from nltk.translate.meteor_score import meteor_score


def corpus_meteor_moses(list_of_refs, list_of_hypos):
    res = []
    for i in range(len(list_of_hypos)):
        refs = [' '.join(ref) for ref in list_of_refs[i]]
        s = meteor_score(refs, ' '.join(list_of_hypos[i]))
        res.append(s)
    return sum(res)/len(res)
