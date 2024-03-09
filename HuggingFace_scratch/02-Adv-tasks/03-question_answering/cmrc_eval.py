import sys
import os
import re

import nltk
tokenize = lambda x: nltk.word_tokenize(x)

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax

def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision     = 1.0*lcs_len/len(prediction_segs)
        recall         = 1.0*lcs_len/len(ans_segs)
        f1             = (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)

def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

# predictions: {example_id: prediction_text}
# references:  {example_id: [answer1, answer2, ...]}
def evaluate_cmrc(predictions, references):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for query_id, answers in references.items():
        total_count += 1
        if query_id not in predictions:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue
        prediction = predictions[query_id]
        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {
        'avg': (em_score + f1_score) * 0.5, 
        'f1': f1_score, 
        'em': em_score, 
        'total': total_count, 
        'skip': skip_count
    }