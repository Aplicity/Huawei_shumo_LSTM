import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd


class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                input_data = np.array(pb_data.get_values()[:,0:17], dtype=np.float32)
                print(file_name, input_data.shape)
                filesDatas.append(input_data)

        filesDatas = np.array(filesDatas,dtype=np.float32).reshape(-1, 17)
        preprocessed_data['myInput'] = filesDatas        
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        return preprocessed_data


    def _postprocess(self, data):        
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output