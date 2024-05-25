import numpy as np
import pickle
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

CLASSES_DEFAULT =  { 
    0: frozenset(),
    1: frozenset(['ptsd']),
    2: frozenset(['depression']),
    3: frozenset(['anxiety']),
    4: frozenset(['depression', 'anxiety']),
    5: frozenset(['depression', 'ptsd']),
    6: frozenset(['anxiety', 'ptsd']),
    7: frozenset(['depression', 'anxiety', 'ptsd']),
}

CLASSES_COMORBID = { 
    4: frozenset(['depression', 'anxiety']),
    5: frozenset(['depression', 'ptsd']),
    6: frozenset(['anxiety', 'ptsd']),
    7: frozenset(['depression', 'anxiety', 'ptsd']),
}

CLASS_HEALTHY = CLASSES_DEFAULT[0]


class ClassEmbedders:
    @staticmethod
    def match_exact(diseases, classes, classes_lookup):
        '''Returns an exact match, diseases == class'''
        diseases_set = frozenset(diseases)
        if diseases_set in classes_lookup:
            return classes_lookup[diseases_set] 
        return None

    @staticmethod
    def match_closest(diseases, classes, classes_lookup):
        '''Returns the closest class match for the diseases by using Jaccard Similarity. Or None if there is no closest match.'''
        exact = ClassEmbedders.match_exact(diseases, classes, classes_lookup)
        if exact != None:
            return exact
        
        closest_match_score = 0
        closest_match_num = None
        for num, class_disease_set in classes.items():
            # calcuate Jaccard Similarity between J(diseases, class_disease_set)
            D = float(len(diseases.union(class_disease_set)))
            if (D > 0):
                match_score = float(len(diseases.intersection(class_disease_set)))/D
                if (match_score > closest_match_score):
                    closest_match_score = match_score
                    closest_match_num = num
        return closest_match_num


class DataPreparationContext:
    def __init__(self, classes=CLASSES_DEFAULT, class_embedder=ClassEmbedders.match_exact, combine_riskiest=16, verbose=False):
        self.classes = classes
        self.class_embedder = class_embedder
        self.combine_riskiest = combine_riskiest
        self.verbose = verbose
        self.classes_lookup = dict((v,k) for k,v in classes.items())

    def print_sample_stats(self, y):
        buckets = {}
        for i in range(len(y)):
            clazz = self.classes[y[i]]
            if not clazz in buckets:
                buckets[clazz] = 0
            buckets[clazz] += 1
        
        # print stats
        for key in buckets:
            print(f" [{' '.join(key)}] has {buckets[key]}")
        print("")
    
    @staticmethod
    def combine_top_risky_symptom_probs(symp_probs, riskiest):
        '''Selects the top riskiest symptom probs. Where riskiest is defined as highest sum of symptom probs'''
        # calculate risky scores
        risky_scores = []
        if len(symp_probs) < riskiest:
            raise Exception("Not enough data!")
        for row in symp_probs:
            risky_scores.append(row)
        # sort by risky
        risky_scores = sorted(risky_scores, reverse=True, key=lambda x:np.sum(x).astype(float))

        # build the result as one big matrix
        result = []
        for i in range(riskiest):
            for elem in risky_scores[i]:
                result.append(elem)
        return result

    def prepare_data(self, raw_data, oversample=False):
        '''Filters non-relevant classes from the dataset'''
        result = {}
        result['classes'] = self.classes
        result['X'] = []
        result['y'] = []
        for src in tqdm(raw_data):
            disease_class = self.class_embedder(frozenset(src['diseases']), self.classes, self.classes_lookup)
            if disease_class != None:
                if self.combine_riskiest <= 0:
                    # don't combine any rows. Output them all separately
                    for row in src['symp_probs']:
                        result['X'].append(row)
                        result['y'].append(disease_class)
                else:
                    if len(src['symp_probs']) < self.combine_riskiest:
                        if self.verbose:
                            print(f"Not enough data! Skipping row. got: {len(src['symp_probs'])} rows, expected: {self.combine_riskiest}")
                    else:
                        # combine the top riskiest
                        combined = self.combine_top_risky_symptom_probs(src['symp_probs'], riskiest=self.combine_riskiest)
                        result['X'].append(combined)
                        result['y'].append(disease_class)
        if oversample:
            ros = RandomOverSampler()
            result['X'], result['y'] = ros.fit_resample(result['X'], result['y'])

        return result

    def prepare_from_file(self, input_filename, oversample=False):
        with open(input_filename, "rb") as f:
            raw_data = pickle.load(f)
            result = self.prepare_data(raw_data, oversample=oversample)
            if self.verbose:
                print(f"Input File: {input_filename}")
                self.print_sample_stats(y=result['y'])
        return result

    def dump_to_file(self, result, output_filename):
        with open(output_filename, "wb") as f:
            pickle.dump(result, f)
        
