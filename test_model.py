import statistics
import numpy
import numpy as np
from blackbox import blackbox

# Die wievielt höchste Wahrscheinlichkeit ist die richtige FA? --> richtiges ergebnis
# 1 = 100
# 2 = 80
# 3 = 50
# 4 = 10
# Gibt es viel Rauschen? --> cofidence
# 1/Varianz subtrahieren

verification_data = [["ensure unsecure ldap is not used", 1],
                     ["ensure rockets are not built on the moon", 0],
                     ["initial passwords must be changed after first use", 27],
                     ["user accounts must be used with least privileges", 28],
                     ]


class BlackboxTester:
    bb = blackbox()
    scores = []  # list to collect all scores
    final_score = None

    def __init__(self):
        self.final_score = self.__get_overall_score(verification_data)

    def __calculate_score(self, verification_string, correct_functional_req):
        score = 0  # initialize score
        # get probability deviation using blackbox
        pred_prob = self.bb.get_prediction_probability_for_sample(verification_string)
        tmp = numpy.copy(pred_prob)
        tmp.sort()
        prob_dict = {}
        # Die 4 höchsten wahrscheinlichkeiten auslesen und entsprechend der korrekten FA Punkte vergeben
        rank = []  # [{'prob': 0.37, 'index': 28}, {'prob': 0.15, 'index': 14}, {'prob': 0.09, 'index': 31}, {'prob': 0.08, 'index': 32}]
        for i in range(1, 5):
            rank.append({"prob": tmp[-i], "index": numpy.where(pred_prob == tmp[-i])[0][0]})
        if rank[0]["index"] == correct_functional_req:  # höchste Wahrscheinlichkeit ist richtige FA
            score = 100
        if rank[1]["index"] == correct_functional_req:  # zweithöchste Wahrscheinlichkeit ist richtige FA
            score = 80
        if rank[2]["index"] == correct_functional_req:  # dritthöchste Wahrscheinlichkeit ist richtige FA
            score = 50
        if rank[3]["index"] == correct_functional_req:  # vierthöchste Wahrscheinlichkeit ist richtige FA
            score = 10
        # Das Rauschen (1/(Varianz * Anzahl FA)) vom Score abziehen
        varianz = statistics.variance(pred_prob)
        subtract = 1 / (varianz * len(pred_prob))
        # negative Scores vermeiden
        score = score - subtract
        if score < 0:
            score = 0
        return score

    def __get_overall_score(self, verification_list):
        for item in verification_data:
            self.scores.append(self.__calculate_score(item[0], item[1]))
        # Gesamtergebnis ist das arithmetische Mittel aller Scores
        final_Score = np.mean(self.scores)
        return final_Score


if __name__ == '__main__':
    bt = BlackboxTester()
    print("[+] Scores = " + str(bt.scores))
    print("[+] Final Score = " + str(bt.final_score))
