import os
import json
import pprint
import importlib
from nupic.swarming import permutations_runner

class Swarmrunner (object):
	'''Runner to swarm for each of field name in the input file'''

	def __init__(self, input_file, output_fpath, field_names, field_types,
		swarmSize, swarm_template, iterationCount, max_workers, input_only_predicted_field):

		'''
		The swarming processes includes permuting over specified meta parameters
		to optimize model predictions.

		:param input_file: csv file with data in nupic format
		:param output_fpath: directory output should be saved to
		:param field_names: list with column header in the data file
		:param field_types: list of columns' data types, eg 'datetime', 'float' or 'int'
		:param swarmSize: can be 'small', 'medium or 'large'
		:param swarm_template: can be a dictionary or a file with json or py extension.
		For the latter, a permutations.py file is assumed for constrained swarming,
		expects "description.py" in the same folder. To get the coorrect folder structure
		Templates for the permutations.py and description.py files to modify, run one of
		the other two options first. Note that setting iterationCount or swarmSize will
		not change these files/ the swarming process!
		'''

		self.output_fpath = output_fpath
		self.swarmSize = swarmSize
		self.iterationCount = iterationCount
		self.max_workers = max_workers
		self.field_names = field_names
		self.field_types = field_types
		self.input_file = input_file
		self.input_only_predicted_field = input_only_predicted_field

		if not os.path.exists(self.output_fpath):
		    Swarmrunner.createDir(os.path.dirname(self.output_fpath))

		#if isinstance(swarm_template, dict):
		self.swarm_desc = swarm_template
		self.swarmDescType = 'template'

		#else:
		try:
			ext = os.path.splitext(swarm_template)[1]
		except AttributeError as err:
			print("AttributeError: {}".format(err))

		else:
			if ext == '.json':
				with open(swarm_template) as fp:
					self.swarm_desc = json.load(fp)
			elif ext == '.py':
				self.swarmDescType = 'permScript'
			else:
			    raise Exception("Invalid swarm file format")

		self.model_params = None


	def run (self):
		'''
		run swarms for each of the field_names
		'''

		if self.swarmDescType == 'template':
			self._update_swarmDesc()

		print self.field_names

		for field_name in self.field_names:
			if field_name != "timestamp":
			    print field_name
			    self.field_output_fpath = os.path.join(self.output_fpath, field_name)

			    if self.swarmDescType == 'template':
			        self._add_predField_to_swarmDesc(field_name)

			    self._runSwarm()

	def _runSwarm (self):

		if self.swarmDescType == 'template':

			pred_field = self.swarm_desc["inferenceArgs"]["predictedField"]
			self.model_params = permutations_runner.runWithConfig(
			                        self.swarm_desc,
			                        {"maxWorkers": self.max_workers, "overwrite": True},
			                        outputLabel=pred_field,
			                        outDir=self.field_output_fpath,
			                        permWorkDir=self.output_fpath,
			                        verbosity=0)

		elif self.swarmDescType == 'permScript':

			swarm_fpath = os.path.join(self.field_output_fpath, self.swarm_desc)
			import_name = os.path.join(self.field_output_fpath, 'description')
			#import_name = os.path.join(os.path.dirname(self.swarm_desc), 'description')
			import_name = import_name.replace('/','.')

			try:
				print import_name
				imported_desc = importlib.import_module(
				import_name).control

			except ImportError as err:
				print("Import error: {}".format(err))

			pred_field = imported_desc["inferenceArgs"]["predictedField"]
			print pred_field

			self.model_params = permutations_runner.runWithPermutationsScript(
									swarm_fpath,
									{"maxWorkers": self.max_workers, "overwrite": True},
									outputLabel=pred_field,
									permWorkDir=self.field_output_fpath)

		self._params_to_file(pred_field)

		return self.model_params_file


	def _update_swarmDesc(self):
		"""create base swarm desc file with all field names"""

		self.swarm_desc["streamDef"]["streams"][0]["source"] = "file://"+ self.input_file
		self.swarm_desc["swarmSize"] = self.swarmSize
		self.swarm_desc["iterationCount"] = self.iterationCount

		# input_fields = []
		# for fn, ft in zip(self.field_names, self.field_types):
		#     #if fn != "timestamp":
		#     input_fields.append({"fieldName": fn, "fieldType": ft})
		#
		# # self.swarm_desc["includedFields"] = input_fields

	def _add_predField_to_swarmDesc(self, pred_field):
		"""create swarm desc file for given field name"""

		if not os.path.exists(self.field_output_fpath):
		    Swarmrunner.createDir(self.field_output_fpath) #


		fpath = os.path.join(self.field_output_fpath, pred_field+".json")
		#new_config = self.swarm_desc
		self.swarm_desc["inferenceArgs"]["predictedField"] = pred_field

		if self.input_only_predicted_field:
			selected_fields = [pred_field]
			input_names, input_types = \
				self.get_field_indices(self.field_names, self.field_types, selected_fields)
		else:
			input_names, input_types = self.field_names, self.field_types


		input_fields = []
		for fn, ft in zip(input_names, input_types):
		    #if fn != "timestamp":
		    input_fields.append({"fieldName": fn, "fieldType": ft})

		self.swarm_desc["includedFields"] = input_fields


		# even if all fields are selected for prediction one can limit the number of input channels here

		with open(fpath, "w") as fp:
		    json.dump(self.swarm_desc, fp, indent = 4)

		return fpath

	def _params_to_file(self, pred_field):

		params_name = "model_params.py"

		self.model_params_file = os.path.join(self.field_output_fpath, params_name)

		with open(self.model_params_file, "wb") as out_file:
		    model_params_string = Swarmrunner.params_to_string(self.model_params)
		    out_file.write("MODEL_PARAMS = \\\n%s" % model_params_string)

	@staticmethod
	def params_to_string(model_params):
		"""
		"""
		pp = pprint.PrettyPrinter(indent=2)
		return pp.pformat(model_params)

	@staticmethod
	def createDir (fpath):
		"""make sure Python finds all subdirectories"""
		fpath = os.path.abspath(fpath)

		while not os.path.exists(os.path.dirname(fpath)):
		    Swarmrunner.createDir (os.path.dirname(fpath))

		print ("creating {}".format(fpath))
		os.makedirs(fpath)
		ff = open(os.path.join(fpath, "__init__.py"), "w")
		ff.close()

	@staticmethod
	def get_field_indices(field_names, field_types, selected_fields):
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
