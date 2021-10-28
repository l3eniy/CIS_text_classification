import statistics
import numpy
import numpy as np
from blackbox import blackbox
import threading
import time

# Die wievielt höchste Wahrscheinlichkeit ist die richtige FA? --> richtiges ergebnis
# 1 = 100
# 2 = 80
# 3 = 50
# 4 = 10
# Gibt es viel Rauschen? --> cofidence
# 1/Varianz subtrahieren

verification_data = [["Ensure IMAP and POP3 server is not installed", 1],
                     ["Ensure all AppArmor Profiles are in enforce or complain mode", 6],
                     ["Ensure sudo log file exists", 33],
                     ["Ensure minimum days between password changes is configured", 31],
                     ["This is no CIS requirement", 0]
                     ]


class BlackboxTester:

    def __init__(self):
        self.bb = blackbox()
        self.scores = []  # list to collect all scores
        self.final_score = None
        self.__get_overall_score(verification_data)

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
        for item in verification_list:
            self.scores.append(self.__calculate_score(item[0], item[1]))
        # Gesamtergebnis ist das arithmetische Mittel aller Scores
        self.final_score = np.mean(self.scores)


if __name__ == '__main__':
    bag_of_scores = []
    objs = [BlackboxTester() for i in range(10)]
    for obj in objs:
        bag_of_scores.append(obj.final_score)
        # print("[+] Scores = " + str(obj.scores))
    print("[+] Overall Score is " + str(np.mean(bag_of_scores)))

    # print("[+] Scores = " + str(obj.scores))
    # print("[+] Final Score = " + str(obj.final_score))