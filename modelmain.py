import os
import sys
import json
import glob
import re
import importlib
import warnings
from swarmrunner import Swarmrunner
from modelrunner import Modelrunner

import numpy as np


class Model(object):


    def __init__(self, input_file, output_fpath,
                 selected_fields=None):

        if output_fpath[-1] != '/':
            output_fpath = output_fpath + '/'

        self.input_file = os.environ['HOME'] + input_file
        self.output_fpath = 'analysis/' + output_fpath
        self.suffix = ''
        self.best_params = False
        self.join_scores = True

        with open(self.input_file) as fp:
            self.field_names = fp.readline().strip("\n").split(",")
            self.field_types = fp.readline().strip("\n").split(",")

        # select field names; account for chosing via index or name


        if selected_fields:
            self.field_names, self.field_types = \
                self.get_field_indices(self.field_names,
                                       self.field_types,
                                       selected_fields)


        # Try to read in model params for given fields
        self._readModelParams()

        # get prefix
        self.prefix = os.path.commonprefix(self.field_names)
        #print ("prefix: {}".format(self.prefix))


    def predict(self, bestParams=False):
        '''
        run the prediction with the parameters in the folder
        '''

        self.best_params = bestParams

        print 'self.field_names: ', self.field_names

        if len(self.field_names) == 1:
            self.join_scores = False

        self.models = []

        # takeout timestamp
        for i, field_name in enumerate(self.field_names):
            if field_name != "timestamp":
                print 'field_name: ', field_name

                self.models.append(self.run_field(self.output_fpath,
                               os.path.join(
                                   self.output_fpath, field_name,
                                   field_name + ".json"),
                               val_idx=i))

        if self.join_scores:
            self.join_score()

        self.cleanup()

    def swarm(self, swarm_template='swarm_description_template.json',
                 swarmSize='small', iterationCount=3000, max_workers = 4, input_only_predicted_field=True):

        self.swarmrunner = Swarmrunner(self.input_file, self.output_fpath,
                               self.field_names, self.field_types,
                               swarmSize=swarmSize, iterationCount=iterationCount,
                               swarm_template=swarm_template, max_workers = max_workers,
                               input_only_predicted_field=input_only_predicted_field)

        self.swarmrunner.run()
        self._readModelParams()

        # TODO: make swarmrunner also save a list of all model_params
        #self.updateModelParams(self.swarmrunner.model_params)

        self.cleanup()

    def _readModelParams (self):

        self.model_params = []

        print 'Adding model params...'
        for field in self.field_names:

            if field != 'timestamp':
                model_fpath = os.path.join(self.output_fpath, field, 'model_params').replace("/", ".")

                try:
                    self.model_params.append(importlib.import_module(
                        model_fpath).MODEL_PARAMS)
                    print model_fpath

                except:
                    print 'could not get model params for: ', model_fpath

    def updateModelParams (self,model_params):

        self.model_params = model_params
        field_names = self.field_names
        if "timestamp" in field_names:
            field_names.remove("timestamp")

        print 'Updating model params...'
        for ix, field in enumerate(field_names):
            model_fpath = os.path.join(self.output_fpath, field, 'model_params.py')
            print model_fpath
            with open(model_fpath, "wb") as out_file:
                model_params_string = Swarmrunner.params_to_string(self.model_params[ix])
                out_file.write("MODEL_PARAMS = \\\n%s" % model_params_string)

    def printModelParams(self):

        for params in self.model_params:
            print json.dumps(params, indent=2)

    def run_field(self, output_fpath, config_fpath, val_idx=0):
        # print ("config path: {}".format(config_fpath))
        print 'config path: ', config_fpath
        with open(config_fpath) as fp:
            swarm_desc = json.load(fp)

        #input_fpath = self.input_file
        input_fpath_swarm = swarm_desc["streamDef"]["streams"][0]["source"].split(
            "file://")[1]

        if self.input_file != input_fpath_swarm:
            warning_message = 'The input file specified in the model differs from the one that has been used for swarming.\ninput_fpath: %s\ninput_fpath_swarm: %s' % (
                self.input_file, input_fpath_swarm)
            warnings.warn(warning_message)

        pred_field = swarm_desc["inferenceArgs"]["predictedField"]

        # work_dir = out_dir
        print pred_field, output_fpath
        # run_model(output_fpath, input_fpath, pred_field, datetime_format, val_idx, swarm, minResolution=minResolution, minVal=minVal, maxVal=maxVal)

        model = Modelrunner(output_fpath, self.input_file,
                            pred_field, val_idx, self.suffix, self.best_params)
        model.run()

        attrs = vars(model)

        print ', '.join("%s: %s" % item for item in attrs.items())

        return model


    def tempIntegrateScore(self, scores, winSize=5, mode='mean'):

        #print scores.shape
        chunks = self.slidingWindow(scores, winSize)

        intScores = []
        for chunk in chunks:

            if mode == 'mean':
                intScores.append(np.mean(chunk))
            elif mode == 'max':
                intScores.append(np.max(chunk))
            elif mode == 'sum':
                intScores.append(np.max(chunk))

        return intScores

    def slidingWindow(self, scores, winSize=5, step=1):

        numOfChunks = (scores.shape[1] / step) + 1

        for i in range(0, numOfChunks * step, step):

            if i < winSize - 1:
                yield scores[:, :i + 1]
            elif i >= winSize:
                yield scores[:, i - winSize:i]

    def join_score(self):
        # print
        # type(glob.glob(self.output_fpath+self.prefix+"[!data|final_lh|Report]*.csv"))

        output_fpath_with_prefix = self.output_fpath + self.prefix

        # All output files for this sensor (exclude combined data files)
        files = \
            list(set(glob.glob(output_fpath_with_prefix + '*' + self.suffix + '.csv')) -
                 set(glob.glob(output_fpath_with_prefix + '*Report*.csv')) -
                 set(glob.glob(output_fpath_with_prefix + '*final_lh*')))

        all_scores = []

        print '\nFiles to be merged to one csv: \n', files, '\n'

        for file in files:
            # print file
            with open(file) as fp:
                # get anomaly, likelihood and log likelihood scores
                scores = map(lambda line: line.strip(
                    "\n").split(",")[-4:-1], fp.readlines()[3:])

            all_scores.append(scores)

        ts = map(lambda line: line.strip("\n").split(
            ",")[0], open(files[0]).readlines()[3:])
        all_scores = np.array(all_scores)
        all_scores = all_scores.astype(np.float)
        #print all_scores.shape

        modes = ['mean','mean','sum']
        intScores = [self.tempIntegrateScore(all_scores[:,:,ix],
            winSize=5, mode=modes[ix]) for ix in range(all_scores.shape[2])]

        #max_lhs = map(lambda tup: max(tup), zip(*all_lhs))

        #intScores = self.tempIntegrateScore(
        #    all_ass.astype(np.float), winSize=5, mode='mean')

        with open(self.output_fpath + self.prefix + "final_lh" + self.suffix + ".csv", "w") as fp:
            # with open("data/"+sensor+"_final_lh.csv", "w") as fp:
            for t, sc, lh, lo in zip(ts, intScores[0], intScores[1], intScores[2]):
                print >> fp, t + "," + str(sc) + "," + str(lh) + "," + str(lo)


    def cleanup(self):

        for dirpath, dirnames, filenames in os.walk(self.output_fpath):
            print 'dirpath: ', dirpath
            for filename in filenames:
                if re.search(r"pyc|pkl|Report", filename):
                    print 'remove %s' % filename
                    os.remove(os.path.join(dirpath, filename))

    @staticmethod
    def get_field_indices(field_names, field_types,
                          selected_fields):
        if len(selected_fields) == len([x for x in selected_fields if type(x) == int]):
            # select fields based on index

            field_names_indices = selected_fields
        elif len(selected_fields) == len([x for x in selected_fields if type(x) == str]):
            # extract indices based on names
            field_names_indices = [i for i, item in enumerate(
            field_names) if item in selected_fields]

        else:
            print 'There is a problems with your field selection. Please make sure to specify selected fields either with indices (int) or names (str), but do not mix. Qutting script.'

            # always select timestamp
        if 'timestamp' in field_names:
            field_names_indices.append(0)
            field_names_indices.sort()


        field_names = [field_names[ix]
                            for ix in field_names_indices]
        field_types = [field_types[ix]
                            for ix in field_names_indices]

        return field_names, field_types
