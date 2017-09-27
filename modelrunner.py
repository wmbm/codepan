import os
import time
import importlib
import csv
import json
import datetime
from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.frameworks.opf.common_models.cluster_params import (
    getScalarMetricWithTimeOfDayAnomalyParams)
from getMinMax import calcMinMax
from swarmrunner import Swarmrunner


class Modelrunner(object):

    def __init__(self, output_fpath, input_file,
                 pred_field, col_ix, suffix, best_params):

        self.input_file = input_file
        self.output_fpath = output_fpath
        self.datetime_format = "%Y-%m-%d %H:%M:%S"
        self.bestParams = best_params
        self.suffix = suffix

        self.pred_field = pred_field
        self.col_ix = col_ix
        self.resolution = None
        self.minVal = None
        self.maxVal = None
        self.model_fpath = None
        self.model_params = None
        self.minResolution = None

        if self.bestParams:

            self.minVal, self.maxVal, self.resolution = calcMinMax(
                self.input_file, self.col_ix)

            if self.minResolution is not None:
                self.resolution = max(self.minResolution, self.resolution)
            # if self.minVal is not None:
            #     self.minVal = minVal
            # if self.maxVal is not None:
            #     self.maxVal = maxVal

            print ("set resolution {}".format(self.resolution))
            print ("min val {}".format(self.minVal))
            print ("max val {}".format(self.maxVal))

        self.iterationCount = 0
        self.output_file = None

        self.create_model()
        print self.model

    def create_model(self):
        """
        Given a model params dictionary, create a CLA Model. Automatically enables
        inference for "pred_field".
        """

        print os.path.abspath(self.output_fpath)

        if not self.bestParams:

            self.model_fpath = os.path.join(
                self.output_fpath, self.pred_field).replace("/", ".")
            self.model_params_name = 'model_params' + self.suffix

            print "Creating model from %s..." % self.model_params_name
            self.get_model_params()

        else:

            self.model_params = getScalarMetricWithTimeOfDayAnomalyParams(
                metricData=[0],
                tmImplementation="cpp",
                minResolution=self.resolution,
                minVal=self.minVal,
                maxVal=self.maxVal)["modelConfig"]

            self.model_params["modelParams"]["sensorParams"]["encoders"] = Modelrunner.setEncoderParams(
                self.model_params["modelParams"]["sensorParams"]["encoders"], self.pred_field)

            model_dir = self.output_fpath + "model_params/"

            if not os.path.exists(os.path.dirname(model_dir)):
                os.makedirs(model_dir)
                ff = open(os.path.join(model_dir, "__init__.py"), "w")
                ff.close()

            with open(model_dir + self.pred_field + "_model_params.py", 'w') as fp:
                json.dump(self.model_params, fp, indent=4)

        self.model = ModelFactory.create(self.model_params)
        self.model.enableInference({"predictedField": self.pred_field})

    def get_model_params(self):
        """
        Assumes a matching model params python module exists within
        the model_params directory and attempts to import it.
        """
        import_name = "%s.%s" % (self.model_fpath.replace("/", "."),
                                 self.model_params_name)

        print "Importing model params from %s" % import_name
        try:
            self.model_params = importlib.import_module(
                import_name).MODEL_PARAMS
        except ImportError:
            raise Exception(
                "No model params exist for '%s'. Run swarm first or manually create file!" % import_name)

        # return imported_model_params

    def run(self):
        """
        Handles looping over the input data and passing each row into the given model
        object, as well as extracting the result object and passing it into an output
        handler.
        """
        self.output_file = self.output_fpath + self.pred_field + \
                self.suffix + ".csv"

        print ("output saved to {}".format(self.output_file))

        # header_row = ['timestamp', pred_field, 'prediction', 'anomaly_score', 'anomaly_likelihood']
        print "IDXx: ", self.col_ix, self.input_file
        ft = open(self.output_file, "wb")
        anomaly_likelihood_helper = anomaly_likelihood.AnomalyLikelihood()
        with open(self.input_file, "rb") as fp:
            headers = fp.readline().strip("\n").split(",")
            field_types = fp.readline().strip("\n").split(",")
            print field_types[1]
            fp.readline()
            rows_ = fp.readlines()
            # print rows_[0].split(",")[output_fpath]
            for row_ in rows_:
                self.iterationCount += 1
                row = row_.strip("\n").split(",")

                # prepare data as input to model
                modelInput = dict(zip(headers, row))
                for field in modelInput:

                    if field == 'timestamp':
                        modelInput['timestamp'] = datetime.datetime.strptime(
                            modelInput['timestamp'], self.datetime_format)
                    else:
                        if field_types[1]=='float':
                            modelInput[field] = float(modelInput[field])
                        elif field_types[1]=='int':
                            modelInput[field] = int(modelInput[field])

                result = self.model.run(modelInput)

                #row = row_.split(",")
                #timestamp = datetime.datetime.strptime(row[0], datetime_format)
                #val = float(row[output_fpath].strip("\n"))
                #result = model.run({"timestamp": timestamp, pred_field: val})

                if self.bestParams:
                    prediction = None
                    confidence = None

                else:
                    prediction = result.inferences["multiStepBestPredictions"][1]
                    confidence = result.inferences["multiStepPredictions"][1][prediction]

                anomaly_score = result.inferences["anomalyScore"]
                anomaly_Likelihood = anomaly_likelihood_helper.anomalyProbability(
                    modelInput[self.pred_field], anomaly_score, modelInput['timestamp'])
                anomaly_logLikelihood = anomaly_likelihood_helper.computeLogLikelihood(
                    anomaly_Likelihood)
                d = [modelInput['timestamp'].strftime(self.datetime_format), str(modelInput[self.pred_field]), str(prediction), str(anomaly_score),
                     str(anomaly_Likelihood), str(anomaly_logLikelihood), str(confidence)]

                if self.iterationCount % 100 == 0:
                    print '#', self.iterationCount
                # print "writing to file..."
                # print ','.join(d)
                print >> ft, ','.join(d)
                ft.flush()

    @staticmethod
    def setEncoderParams(encoderParams, pred_field):

        encoderParams["timestamp_dayOfWeek"] = encoderParams.pop(
            "c0_dayOfWeek")
        encoderParams["timestamp_timeOfDay"] = encoderParams.pop(
            "c0_timeOfDay")
        encoderParams["timestamp_timeOfDay"]["fieldname"] = "timestamp"
        encoderParams["timestamp_timeOfDay"]["name"] = "timestamp"
        encoderParams["timestamp_weekend"] = encoderParams.pop("c0_weekend")
        encoderParams["value"] = encoderParams.pop("c1")
        encoderParams["value"]["fieldname"] = pred_field
        encoderParams["value"]["name"] = pred_field

        # print encoderParams

        return encoderParams
