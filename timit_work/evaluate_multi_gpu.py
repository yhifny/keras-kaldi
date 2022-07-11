#based on https://github.com/yuanyuanli85/Keras-Multiple-Process-Prediction
from multiprocessing import Process
import argparse
import numpy as np
import os
from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader
import config_kaldi
from numpy import newaxis
from wer import wers_timit39

from keras.models import Model, load_model
from gated_cnn import  GatedConvBlock
from GCNN import GatedConv1D
from novograd import NovoGrad

def predict(models, feats):
    
    # one model    
    if type(models) is not list:
        return models.predict(feats)[0,:,:] 

    if len(models) ==1:
        return models[0].predict(feats)[0,:,:] 


    # posterior fusion
    pred_stack = np.empty((0, len(models)), float)
    pred_list=[]
    for model in models:
        pred=model.predict(feats)[0,:,:]
        pred_list.append(pred)
    mean_pred=np.rollaxis(np.dstack(pred_list),-1).mean(axis=0)
    return mean_pred



def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)

def load_decoder():
    # Construct recognizer
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = 13
    decoder_opts.max_active = 7000
    asr = MappedLatticeFasterRecognizer.from_files(
        config_kaldi.final_model, config_kaldi.graph_file, config_kaldi.words_mapping_file,
        acoustic_scale=1.0, decoder_opts=decoder_opts)
    return asr 

class KerasWorker(Process):
    def __init__(self, gpuid, models, priors, feat_norm_file, feat_list):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid          = gpuid
        self._models          = models
        self._feat_norm_file = feat_norm_file
        self._priors         = priors
        self._feat_list      = feat_list
        self._asr            = load_decoder()
        self.hyp_list        = []
        

    def run(self):

        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        feats_mean  = np.load(self._feat_norm_file)['mean']
        feats_std   = np.load(self._feat_norm_file)['std']

        models=[] 
        for  filename in self._models:
            print (filename)
            kalid_model = load_model(filename, custom_objects={'GatedConv1D':GatedConv1D,'GatedConvBlock':GatedConvBlock, 'NovoGrad': NovoGrad})
            model = Model(inputs=[kalid_model.input[0] if type(kalid_model.input) is list else              kalid_model.input], outputs=[kalid_model.get_layer('output_tri').output])
            model.summary()
            models.append(model)


        for (fkey, feats)   in self._feat_list:
            print ('processing: ', fkey, ' on woker', self._gpuid, flush=True)        	
            feats=normalize(feats.numpy(), feats_mean, feats_std)[newaxis,...]
            loglikes = np.log (predict (models, feats) / self._priors)
            loglikes [loglikes == -np.inf] = -100        
            out = self._asr.decode(Matrix(loglikes))
            self.hyp_list.append("%s %s" %(fkey, out["text"]))



class KerasScheduler:
    def __init__(self, gpuids, models, priors, dataset, feat_norm_file):
        self._gpuids = gpuids
        
        # load data set
        feats_rspecifier = config_kaldi.fmllr_dev_feats_rspecifier
        if dataset == 'test': feats_rspecifier = config_kaldi.fmllr_test_feats_rspecifier
        feat_list = []
        with SequentialMatrixReader(feats_rspecifier) as f:
            for (fkey, feats)   in f:
                feat_list.append((fkey, feats))
        
        # set the workers
        self.__init_workers(models, priors, feat_norm_file, feat_list)

    def __init_workers(self, models,priors, feat_norm_file, feat_list):
        
        feat_list_splited  =   np.array_split(feat_list, len(self._gpuids))
        self._workers      = []
        for i, gpuid in enumerate(self._gpuids):
            self._workers.append(KerasWorker(gpuid, models, priors, 
                                                feat_norm_file, feat_list_splited[i]))


    def start(self):

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        results_list = []
        for worker in self._workers:
            worker.join()
            results_list.extend(hyp_list)
        
        print("all of workers have been done")
        
        return hyp_list
        
        
def evaluate_multi_gpu( gpuids, models, priors, dataset, feat_norm_file, ref_list):
    
    x = KerasScheduler(gpuids, models, priors, dataset, feat_norm_file)        
    hyp_list = x.start()
    return wers_timit39(ref_list, hyp_list)[1]* 100.0, hyp_list
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset to be proceed (dev or test)")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )
    parser.add_argument("--ref_file",  type=str, help="ref text in kaldi format" )
    parser.add_argument("--priors",  type=str, help="priors file" )
    parser.add_argument("--feat_norm_file",  type=str, help="feat norm file" )
    parser.add_argument("--models", nargs='*', help="models to load")
    parser.add_argument("--hyp_output", type=str, help = "hyp output")
    
    args = parser.parse_args()
    
    gpuids = [int(x) for x in args.gpuids.strip().split(',')]
    
    ref_list= open(args.ref_file).readlines()

    models= args.models

    feat_norm_file = args.feat_norm_file

    # read priors
    priors = np.genfromtxt (args.priors, delimiter=',')

    wer_result, hyp_list = evaluate_multi_gpu( gpuids, models, args.priors, args.dataset, 
                                        feat_norm_file, 
                                        ref_list)
    out_file = open(args.hyp_output, "w")
    for hyp in hyp_list:
        out_file.write("%s\n" %(hyp.strip()))
    
    print ("wer:", wer_result)
    