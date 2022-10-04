from utils import load_data
from snorkel.preprocess import preprocessor
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds
from snorkel.labeling import filter_unlabeled_dataframe

from preprocessors import get_person_text
from preprocessors import get_left_tokens
from preprocessors import get_person_last_names

# classes
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

# load data
((df_dev, Y_dev), df_train, (df_test, Y_test)) = load_data()

# just check one data sample
candidate = df_dev.loc[2]
person_names = get_person_text(candidate).person_names
print("Sentence: ", candidate["sentence"])
print("Person 1: ", person_names[0])
print("Person 2: ", person_names[1])

# data preprocessing for enhancing the features
@preprocessor()
def get_text_between(cand):
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.text_between = " ".join(cand.tokens[start:end])
    return cand


# define some information maybe useful
spouses = {"spouse", "wife", "husband", "ex-wife", "ex-husband"}
family = {
    "father", "mother", "sister", "brother", "son", "daughter",
    "grandfather", "grandmother", "uncle", "aunt", "cousin",
    "papa", "baba", "mama", "mom",
}
family = family.union({f + "-in-law" for f in family})
other = {
    "boyfriend", "girlfriend", "boss", "employee", 
    "secretary", "co-worker", "supervisor", "mentor",
    "leader"
}

@labeling_function()
def lf_married(x):
    '''
    Very simple labeling function
    '''
    return POSITIVE if "married" in x.between_tokens else ABSTAIN

@labeling_function(resources=dict(spouses=spouses))
def lf_husband_wife(x, spouses):
    '''
    Simple labeling function with extra spouse resources for assistance
    '''
    # which means, if any word between P1 and P2 shows in the `spouses` set/list
    # this is a POSITIVE case (P1 and P2 are couple)
    # else: not sure, just abstain
    return POSITIVE if len(spouses.intersection(set(x.between_tokens))) > 0 else ABSTAIN

@labeling_function(resources=dict(family=family))
def lf_familial_relationship(x, family):
    '''
    Simple labeling function with extra family information
    '''
    return NEGATIVE if len(family.intersection(set(x.between_tokens))) > 0 else ABSTAIN

@labeling_function(resources=dict(other=other))
def lf_other_relationship(x, other):
    '''
    Simple labeling function with extra other rel information
    '''
    return NEGATIVE if len(other.intersection(set(x.between_tokens))) > 0 else ABSTAIN

# Check for the `spouse` words appearing to the left of the person mentions
@labeling_function(resources=dict(spouses=spouses), pre=[get_left_tokens])
def lf_husband_wife_left_window(x, spouses):
    '''
    A little bit complex labeling function with extra res and data preprocessing
    '''
    if len(set(spouses).intersection(set(x.person1_left_tokens))) > 0:
        return POSITIVE
    elif len(set(spouses).intersection(set(x.person2_left_tokens))) > 0:
        return POSITIVE
    else:
        return ABSTAIN

# create labeling applier
lfs = [
    lf_husband_wife,
    lf_husband_wife_left_window,
    lf_married,
    lf_familial_relationship,
    lf_other_relationship,
]
applier = PandasLFApplier(lfs)

print('* applying dev dataset ...')
L_dev = applier.apply(df_dev)

print('* applying train dataset ...')
L_train = applier.apply(df_train)

# analysis
summary = LFAnalysis(L_dev, lfs).lf_summary(Y_dev)
print("* dev summary", summary)

# create label_model
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, Y_dev, n_epochs=5000, log_freq=500, seed=12345)