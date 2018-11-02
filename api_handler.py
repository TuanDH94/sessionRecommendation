import logging
from lstm_model.predictor import Predictor
import json
import tensorflow as tf

logging.basicConfig(filename='example.log', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')

global graph
graph = tf.get_default_graph()


class ApiHandler:
    predictor = None

    def __init__(self):
        self.predictor = Predictor()

    def get_recommend_output(self, inputs):
        logging.warning(inputs)
        inputs_model = self.parse_input_list(inputs)
        with graph.as_default():
            output, output_value = self.predictor.predict(inputs_model)
        out_map = {}
        for i in range(len(inputs_model)):
            out_map[i] = self.predictor.map_id_to_object(inputs_model[i])
        out_map['output'] = output
        out_map['output_value'] = output_value

        return str(out_map)

    def parse_input_list(self, inputs):
        inputs_model = []
        for input1 in inputs:
            json_data = json.loads(input1)
            if 'pi' in json_data:
                id = json_data['pi']
                if id != '0':
                    inputs_model.append('p_' + str(id))
                    continue
            if 'si' in json_data:
                id = json_data['si']
                if id != '0':
                    inputs_model.append('s_' + str(id))
                    continue
            if 'wi' in json_data:
                id = json_data['wi']
                if id != '0':
                    inputs_model.append('w_' + str(id))
                    continue
            if 'di' in json_data:
                id = json_data['di']
                if id != '0':
                    inputs_model.append('d_' + str(id))
                    continue
        return inputs_model
