import sys
from DynamicClustering import *
from EvolutionaryClustering_original import *
import time


class SituationClustering():
    def __init__(self, input_path, output_path, stop_point=0, model_name=''):
        self.output_path = output_path
        self.stop_point = stop_point
        try:
            self.input_file = open(input_path, 'r')
        except IOError:
            print("Error: Cannot find the input file or the input file is not readable.")
        try:
            if model_name == 'DynamicClustering':
                self.model = DynamicClustering(self.output_path)
            elif model_name == 'EvolutionaryClustering':
                self.model = EvolutionaryClustering_original(self.output_path)
        except IOError:
            print("Error: Please select a situation clustering model.")

    def run(self):
        """
        The main function, including three frames: read_json, update_kb, kb_to_situation_clustering
        """
        count = 0
        start_time = time.time()
        while True:
            line = self.input_file.readline()
            count += 1
            if line:
                dic = json.loads(line)
                self.model.update_kb(dic)
            else:
                break
            if count == self.stop_point:
                break
        self.input_file.close()
        self.model.kb_to_situation_clustering()
        print("--- Running time %s seconds ---" % (time.time() - start_time))

# might need to be improved
input_value = ''
output_value = ''
stopPoint = 0
model_name = ''
if len(sys.argv) >= 3:
    if sys.argv[1] == '-input' and sys.argv[2]:
        input_value = sys.argv[2]
    if len(sys.argv) >= 5:
        if sys.argv[3] == '-output' and sys.argv[4]:
            output_value = sys.argv[4]
            if len(sys.argv) >= 7:
                if sys.argv[5] == '-stop_point' and sys.argv[6]:
                    stop_point = sys.argv[6]
                else:
                    if sys.argv[5] == '-model_name' and sys.argv[6]:
                        model_name = sys.argv[6]
                    else:
                        print('Please specify the command for "-stop_point"')
                if len(sys.argv) >= 9:
                    if sys.argv[7] == '-model_name' and sys.argv[8]:
                        model_name = sys.argv[8]
                    else:
                        print('Please specify the Clustering model.')


SC = SituationClustering(input_path=input_value, output_path=output_value, stop_point=stopPoint, model_name=model_name)
SC.run()
