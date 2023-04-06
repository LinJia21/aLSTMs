import time
import config
import utils
import os
import numpy as np


class Movie2Caption(object):
            
    def __init__(self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,
                 n_frames=None, outof=None
                 ):
        self.signature = signature
        self.model_type = model_type
        self.video_feature = video_feature
        self.maxlen = maxlen
        self.n_words = n_words
        self.K = n_frames
        self.OutOf = outof

        self.mb_size_train = mb_size_train
        self.mb_size_test = mb_size_test
        self.non_pickable = []
        
        self.load_data()
    #_filter_googlenet 方法根据给定的视频 ID 加载特征文件，并使用 get_sub_frames 方法从特征中提取子帧。然后，它将提取的子帧赋值给变量 y，并在后面的代码中返回
    #_filter_googlenet 的介绍网址https://zhuanlan.zhihu.com/p/185025947
    def _filter_googlenet(self, vidID):
        feat_file = os.path.join(self.FEAT_ROOT, vidID + '.npy')
        feat = np.load(feat_file)
        feat = self.get_sub_frames(feat)
        return feat

    # 此方法的目的是获取给定视频的特征。
    def get_video_features(self, vidID):
        if self.video_feature == 'googlenet':
            y = self._filter_googlenet(vidID)
        else:
            raise NotImplementedError()
        return y

    def pad_frames(self, frames, limit, jpegs):
        # pad frames with 0, compatible with both conv and fully connected layers
        last_frame = frames[-1]
        if jpegs:
            frames_padded = frames + [last_frame]*(limit-len(frames))
        else:
            padding = np.asarray([last_frame * 0.]*(limit-len(frames)))
            frames_padded = np.concatenate([frames, padding], axis=0)
        return frames_padded
    
    def extract_frames_equally_spaced(self, frames, how_many):
        # chunk frames into 'how_many' segments and use the first frame
        # from each segment
        n_frames = len(frames)
        splits = np.array_split(range(n_frames), self.K)
        idx_taken = [s[0] for s in splits]
        sub_frames = frames[idx_taken]
        return sub_frames
    
# 此方法的目的是在输入帧中添加视频结束帧。视频结束帧是一个零数组，
# 其形状与输入帧中的最后一帧相同，只是其所有值均为-1。然后，该方法
# 沿着第一个轴将此视频结束帧连接到输入帧上，并返回结果数组。
# 在输入帧中添加视频结束帧的目的可能是为了在视频处理过程中标记视频的结束。
# 这样，在处理视频帧时，程序可以检测到这个特殊的帧，并据此执行相应的操作，
# 例如停止处理或执行其他任务。但是，具体原因取决于程序的实现和使用场景。
    def add_end_of_video_frame(self, frames):
        if len(frames.shape) == 4:
            # feat from conv layer
            _,a,b,c = frames.shape
            eos = np.zeros((1,a,b,c),dtype='float32') - 1.
        elif len(frames.shape) == 2:
            # feat from full connected layer
            _,b = frames.shape
            eos = np.zeros((1,b),dtype='float32') - 1.
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        frames = np.concatenate([frames, eos], axis=0)
        return frames

#             get_sub_frames 方法是 Movie2Caption 类的一个方法，它接受两个参数：frames 和 jpegs。此方法的目的是从输入帧中提取子帧。
#             它首先检查 self.OutOf 是否为 None。如果是，则根据输入帧的长度执行不同的操作。如果输入帧的长度小于 self.K，则使用
#             pad_frames 方法对输入帧进行填充，使其长度达到 self.K。否则，使用 extract_frames_equally_spaced 方法从输入帧中提
#             取 self.K 个子帧。最后，如果 jpegs 为 True，则将结果转换为 numpy 数组并返回
    def get_sub_frames(self, frames, jpegs=False):
        # from all frames, take K of them, then add end of video frame
        # jpegs: to be compatible with visualizations
        if self.OutOf:
            raise NotImplementedError('OutOf has to be None')
            # frames[:self.OutOf]用于从输入帧中提取子帧。如果 self.OutOf 不为 None，则此方法将从输入帧的开头提取 
#             self.OutOf 个帧。这是通过使用切片操作 frames[:self.OutOf] 来实现的，
#             它返回输入帧中从开头到索引 self.OutOf（不包括）的所有帧。  
            frames_ = frames[:self.OutOf]
            if len(frames_) < self.OutOf:
                frames_ = self.pad_frames(frames_, self.OutOf, jpegs)
        else:
            if len(frames) < self.K:
                #frames_ = self.add_end_of_video_frame(frames)
                frames_ = self.pad_frames(frames, self.K, jpegs)
            else:
# extract_frames_equally_spaced 方法是 Movie2Caption 类的一个方法，它接受两个参数：frames 和 how_many。此方法的目的是
# 从输入帧中提取子帧。它首先计算输入帧的数量，然后使用 np.array_split 函数将范围从 0 到输入帧数量的整数序列分成 self.K 个段。
# 然后，它获取每个段的第一个元素的索引，并使用这些索引从输入帧中提取子帧。最后，返回提取的子帧。
                frames_ = self.extract_frames_equally_spaced(frames, self.K)
                #frames_ = self.add_end_of_video_frame(frames_)
        if jpegs:
            frames_ = numpy.asarray(frames_)
        return frames_

    def prepare_data_for_blue(self, whichset):
        # assume one-to-one mapping between ids and features
        feats = []
        feats_mask = []
        if whichset == 'valid':
            ids = self.valid_ids
        elif whichset == 'test':
            ids = self.test_ids
        elif whichset == 'train':
            ids = self.train_ids
        for i, vidID in enumerate(ids):
            feat = self.get_video_features(vidID)
            feats.append(feat)
            feat_mask = self.get_ctx_mask(feat)
            feats_mask.append(feat_mask)
        return feats, feats_mask
    
    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        
        return rval
# The load_data method is a method of the Movie2Caption class. Its purpose is to load data for the Movie2Caption object. It loads train, 
# valid, and test data from the specified dataset path and stores them in the object’s train, valid, and test attributes. It also loads 
# the captions from the dataset path and stores them in the object’s CAP attribute. Additionally, it loads the word dictionary from the 
# dataset path and stores it in the object’s word_ix attribute. The method also sets several other attributes of the object based on the 
# loaded data and the object’s signature and video feature.
    def load_data(self):

        print 'loading %s %s features'%(self.signature, self.video_feature)
        dataset_path = config.RAB_DATASET_BASE_PATH
        self.train = utils.load_pkl(dataset_path + 'train.pkl')
        self.valid = utils.load_pkl(dataset_path + 'valid.pkl')
        self.test = utils.load_pkl(dataset_path + 'test.pkl')
        self.CAP = utils.load_pkl(dataset_path + 'CAP.pkl')
        self.FEAT_ROOT = config.RAB_FEATURE_BASE_PATH
        if self.signature == 'youtube2text':
            self.train_ids = ['vid%s'%i for i in range(1,1201)]
            self.valid_ids = ['vid%s'%i for i in range(1201,1301)]
            self.test_ids = ['vid%s'%i for i in range(1301,1971)]
        elif self.signature == 'msr-vtt':
            self.train_ids = ['video%s'%i for i in range(0,6513)]
            self.valid_ids = ['video%s'%i for i in range(6513,7010)]
            self.test_ids = ['video%s'%i for i in range(7010,10000)]
        else:
            raise NotImplementedError()

        self.word_ix = utils.load_pkl(dataset_path + 'worddict.pkl')
        self.ix_word = dict()
        # word_ix start with index 2
        for kk, vv in self.word_ix.iteritems():
            self.ix_word[vv] = kk
        self.ix_word[0] = '<eos>'
        self.ix_word[1] = 'UNK'

        if self.signature == 'msr-vtt':
            self.n_words = 25000
        if self.video_feature == 'googlenet':
            self.ctx_dim = 2048
        else:
            raise NotImplementedError()
        self.kf_train = utils.generate_minibatch_idx(
            len(self.train), self.mb_size_train)
        self.kf_valid = utils.generate_minibatch_idx(
            len(self.valid), self.mb_size_test)
        self.kf_test = utils.generate_minibatch_idx(
            len(self.test), self.mb_size_test)


def prepare_data(engine, IDs):
    seqs = []
    feat_list = []

    def get_words(vidID, capID):
        caps = engine.CAP[vidID]
        rval = None
        for cap in caps:
            if str(cap['cap_id']) == capID:
                rval = cap['tokenized'].split(' ')
                rval = [w for w in rval if w != '']
                break
        assert rval is not None
        return rval

    for i, ID in enumerate(IDs):
        # load GNet feature
        vidID, capID = ID.split('_')
        feat = engine.get_video_features(vidID)
        feat_list.append(feat)
        words = get_words(vidID, capID)
        seqs.append([engine.word_ix[w]
                     if w in engine.word_ix else 1 for w in words])

    lengths = [len(s) for s in seqs]
    if engine.maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        new_caps = []
        for l, s, y, c in zip(lengths, seqs, feat_list, IDs):
            # sequences that have length >= maxlen will be thrown away
            if l < engine.maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
                new_caps.append(c)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None, None, None

    y = np.asarray(feat_list)
    y_mask = engine.get_ctx_mask(y)
    n_samples = len(seqs)
    maxlen = np.max(lengths)+1

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y, y_mask


def test_data_engine():
    from sklearn.cross_validation import KFold
    video_feature = 'googlenet' 
    out_of = None
    maxlen = 100
    mb_size_train = 64
    mb_size_test = 128
    maxlen = 50
    n_words = 30000 # 25770 
    signature = 'youtube2text' #'youtube2text'
    engine = Movie2Caption('attention', signature, video_feature,
                           mb_size_train, mb_size_test, maxlen,
                           n_words,
                           n_frames=26,
                           outof=out_of)
    i = 0
    t = time.time()
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train[index] for index in idx]
        x, mask, ctx, ctx_mask = prepare_data(engine, ids)
        print 'seen %d minibatches, used time %.2f '%(i,time.time()-t0)
        if i == 10:
            break

    print 'used time %.2f'%(time.time()-t)
if __name__ == '__main__':
    test_data_engine()


