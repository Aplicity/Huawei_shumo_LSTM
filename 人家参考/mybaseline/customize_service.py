import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
from dataProcess import add_clutter,calculate_attr


class mnist_service(TfServingBaseService):

	def _preprocess(self, data):
		preprocessed_data = {}
		filesDatas = []
		print("***begin_read_csv**")
		for k, v in data.items():
			for file_name, file_content in v.items():
				one_sheet = pd.read_csv(file_content)
				one_sheet.fillna(0,inplace = True)
				one_x_sheet =  one_sheet[['Electrical Downtilt','Mechanical Downtilt','RS Power']]
				one_x_sheet = one_x_sheet.join(add_clutter(one_sheet))
				one_sheet['n'] = one_x_sheet['n']
				one_x_sheet = one_x_sheet.join(calculate_attr(one_sheet))
				one_x_sheet.drop([1,3,4,9,19,20],axis=1,inplace = True)
				one_x_sheet = (one_x_sheet - one_x_sheet.mean())/one_x_sheet.std() 
				one_x_sheet.fillna(0,inplace = True)
				len_data = len(one_x_sheet.columns)
				one_x_sheet = np.array(one_x_sheet.get_values()[:,0:len_data], dtype=np.float32)
				print(file_name, one_x_sheet.shape)
				filesDatas.extend(one_x_sheet)
		filesDatas = np.array(filesDatas,dtype=np.float32).reshape(-1, len_data)
		preprocessed_data['myInput'] = filesDatas
		print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
		print("***end_read_csv**")
		return preprocessed_data


	def _postprocess(self, data):
		infer_output = {"RSRP": []}
		print(data)
		for output_name, results in data.items():
			print(output_name, np.array(results).shape)
			infer_output["RSRP"] = results
			print('results')
		return infer_output
