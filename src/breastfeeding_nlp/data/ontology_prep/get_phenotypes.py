# ALL CODE DIRECTLY COPIED FROM CLINPHEN EXCEPT FOR MODIFICATIONS
# MODIFICATIONS MADE: 
# *output table has all individual occurrences of each term
# *term occurrences are tracked by (term_id,index) pairs
# *output includes indicators for context
# *output includes HPO synonyms in the text
# *flag word lists are now text files instead of lists in this script
# *have ID_TO_SYN_MAP_FILE (tap separated lines of HPO_ID -> synonym) and ID_TO_NAME_MAP_FILE (tab sep hpoID -> name), and functions for loading these as dicts
# *removed part of return line of function "load_medical_record_subsentences"; this appear to cause lines with colons to be read twice
# *added "extract_phenotypes_df" for pandas DataFrame version of output
# in "string_to_record_nonlinewise" inserted placeholder "\t" characters to force "end-of-point" breaks
    ### avoids situations like
    ### "head/face/neck: negative"
    ### "eyes: eye abnormality"
    ###
    ### being tokenized as "negative eyes: eye abnormality" --> negative marker on "eye abnormality"

## use Kevin Wu's improvement to avoid load WordNetLemmatizer on each call
##  (see https://bitbucket.org/bejerano/clinphen/branch/feature/performance-improvements)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from collections import defaultdict,Counter
import itertools

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

import pandas as pd

#assert os.path.isfile(DEFAULT_HPO_SYN_MAP_FILE), "Cannot find HPO syn file (have you run GET_STARTED?): {}".format(DEFAULT_HPO_SYN_MAP_FILE)
BASE_DIR = os.path.abspath("../src/breastfeeding_nlp/data/ontology_prep")
ID_TO_SYN_MAP_FILE = f"{BASE_DIR}/hpo_synonyms.txt"
ID_TO_NAME_MAP_FILE = f"{BASE_DIR}/hpo_term_names.txt"
PHENOTYPIC_ABNORMALITY_ID = "FPO:0000001"

def load_hpo_names(filename = ID_TO_NAME_MAP_FILE):
  with open(filename,'r') as f:
    id_to_name =  {
      x:name for x,name in list(
        map(
          lambda x:x.strip('\n').split('\t'),
          f.readlines()
          )
        )
      }
    return id_to_name

#Returns a map from an HPO ID to the full list of its synonymous names
def load_all_hpo_synonyms(filename=ID_TO_SYN_MAP_FILE):
  returnMap = defaultdict(set)
  for line in open(filename):
    lineData = line.strip().split("\t")
    hpo = lineData[0]
    syn = lineData[1]
    returnMap[hpo].add(syn)
  return returnMap

ID_TO_NAME_MAP = load_hpo_names()
ID_TO_SYN_MAP = load_all_hpo_synonyms()



LEMMATIZER = WordNetLemmatizer() ## From Kevin Wu

#### Word/sentence tokenization and lemmatization
def lemmatize(word):
  word = re.sub('[^0-9a-zA-Z]+', '', word)
  word = word.lower()
  #return WordNetLemmatizer().lemmatize(word)
  return LEMMATIZER.lemmatize(word)


#Does the given word imply that, as far as we are concerned, the sentence is over?
#point_enders = [".", ":"]
point_enders = [".", u'•', '•', ";", "\t",]
def end_of_point(word):
  #for char in point_enders:
  #  if char in word: return True
  if word[-1] in point_enders: return True
  if word == "but": return True
  if word == "except": return True
  if word == "however": return True
  if word == "though": return True
  return False

#Parses the medical record into a list of sentences.
subpoint_enders = [",",":"]
def end_of_subpoint(word):
  if word[-1] in subpoint_enders: return True
  if word == "and": return True
  return False

def string_to_record_linewise(medical_record):
  return medical_record.split("\n")

def load_medical_record_linewise(medical_record): #processes lines without colons
  recordFile = string_to_record_linewise(medical_record)
  sentences = []
  for line in recordFile:
    if ":" not in line: continue
    curSentence = []
    for word in line.strip().split(" "):
      word = word.lower()
      if len(word) < 1: continue
      curSentence.append(word)
      if end_of_point(word):
        sentences.append(" ".join(curSentence))
        curSentence = []
    if len(curSentence) > 0: sentences.append(" ".join(curSentence))
  subsentence_sets = []
  for sent in sentences:
    subsents = []
    curSubsent = []
    for word in sent.split(" "):
      word = word.lower()
      curSubsent.append(word)
      if end_of_subpoint(word):
        subsents.append(" ".join(curSubsent))
        curSubsent = []
    if len(curSubsent) > 0: subsents.append(" ".join(curSubsent))
    subsentence_sets.append(subsents)
  return subsentence_sets

def string_to_record_nonlinewise(medical_record):
  listForm = []
  #for line in medical_record.replace('\n','\t\n').split("\n"): ### added ".replace('\n','\t\n')" to make line breaks end-of-point breaks"
  for line in medical_record.split("\n"):
    if len(line) < 1: continue
    listForm.append(line.strip('\t'))
  return " ".join(listForm).split(" ")

def load_medical_record_subsentences(medical_record):
  record = string_to_record_nonlinewise(medical_record)
  sentences = []
  curSentence = []
  for word in record:
    word = word.lower()
    if len(word) < 1: continue
    curSentence.append(word)
    if end_of_point(word):
      sentences.append(" ".join(curSentence))
      curSentence = []
  if len(curSentence) > 0: sentences.append(" ".join(curSentence))
  subsentence_sets = []
  for sent in sentences:
    subsents = []
    curSubsent = []
    for word in sent.split(" "):
      word = word.lower()
      curSubsent.append(word)
      if end_of_subpoint(word):
        subsents.append(" ".join(curSubsent))
        curSubsent = []
    if len(curSubsent) > 0: subsents.append(" ".join(curSubsent))
    subsentence_sets.append(subsents)
  #return subsentence_sets + load_medical_record_linewise(medical_record) ### duplicates lines with colons: linewise hits all, nonlinewise captures sentences with colons
  return load_medical_record_linewise(medical_record)

def load_mr_map(parsed_record):
  returnMap = defaultdict(set)
  for i in range(len(parsed_record)):
    line = set(parsed_record[i])
    for word in line: returnMap[word].add(i)
  return returnMap

#Checks the given sentence for any flags from the lists you indicate.
######

# FLAGS = {
#   'negative':[],
#   'family':[],
#   'healthy':[],
#   'disease':[],
# #  'treatment':treatment_flags,
# #  'history':history_flags,
# #  'mild':mild_flags,
# #  'uncertain':[]
# }
# for f in FLAGS:
#   with open(os.path.join(os.path.dirname(__file__), f"data/flags/{f}.txt"),'r') as fil:
#     lines = fil.read().lower()
#     lines = re.sub("#[^\n]*\n","\n",lines)
#     FLAGS[f] = lines.split('\n')


# low_synonyms = set(["low", "decreased", "decrease", "deficient", "deficiency", "deficit", "deficits", "reduce", "reduced", "lack", "lacking", "insufficient", "impairment", "impaired", "impair", "difficulty", "difficulties", "trouble"])
# high_synonyms = set(["high", "increased", "increase", "elevated", "elevate", "elevation"])
# abnormal_synonyms = set(["abnormal", "unusual", "atypical", "abnormality", "anomaly", "anomalies", "problem"])
# common_synonyms = [
# low_synonyms,
# high_synonyms,
# abnormal_synonyms
# ]

def synonym_lemmas(word):
  returnSet = set()
  # for synSet in common_synonyms:
  #   if word in synSet: returnSet |= synSet
  return returnSet

def custom_lemmas(word):
  returnSet = set()
  if len(word) < 2: return returnSet
  if word[-1] == "s": returnSet.add(word[:-1])
  if word[-1] == "i": returnSet.add(word[:-1] + "us")
  if word [-1] == "a":
    returnSet.add(word[:-1] + "um")
    returnSet.add(word[:-1] + "on")
  if len(word) < 3: return returnSet
  if (word[-2:] == "es") & (word[:-2].lower() != "not"):    # not lemmatizing "notes" to "not"
    returnSet.add(word[:-2])
    returnSet.add(word[:-2] + "is")
  if word[-2:] == "ic":
    returnSet.add(word[:-2] + "ia")
    returnSet.add(word[:-2] + "y")
  if word[-2:] == "ly": returnSet.add(word[:-2])
  if (word[-2:] == "ed") & (word[:-2].lower() != "not"): returnSet.add(word[:-2])    # not lemmatizing "noted" to "not"
  if len(word) < 4: return returnSet
  if word[-3:] == "ata": returnSet.add(word[:-2])
  if word[-3:] == "ies": returnSet.add(word[:-3] + "y")
  if word[-3:] == "ble": returnSet.add(word[:-2] + "ility")
  if len(word) < 7: return returnSet
  if word[-6:] == "bility": returnSet.add(word[:-5] + "le")
  if len(word) < 8: return returnSet
  if word[-7:] == "ication":
    returnSet.add(word[:-7] + "y")
    returnSet.add(word[:-7] + "ied")
  return returnSet


def add_lemmas(wordSet):
  lemmas = set()
  for word in wordSet:
    lemma = lemmatize(word)
    if len(lemma) > 0: lemmas.add(lemma)
    lemmas |= synonym_lemmas(word)
    lemmas |= custom_lemmas(word)
  return wordSet | lemmas


# FLAG_TYPE = dict()
# for f in FLAGS:
#   for w in add_lemmas(set(FLAGS[f])):
#     FLAG_TYPE[w] = f

#def get_flags(line, *flagsets):
#  line = add_lemmas(set(line))
#  returnFlags = []
#  for flagset in flagsets:
#    flagset = add_lemmas(set(flagset))
#    for word in flagset:
#      if word in line: returnFlags.append(FLAG_TYPE[word])
#  return returnFlags

# def get_flags(line, flag_dict = FLAGS, ignore_tokens = {}):
#   line = add_lemmas(set(line))
#   returnFlags = []
#   for flag_type in flag_dict:
#     flagset = add_lemmas(set(flag_dict[flag_type]))
#     for word in flagset:
#       if word in line: returnFlags.append(word)
#   return returnFlags


#def alphanum_only(wordSet):
#  returnSet = set()
#  for word in wordSet:
#    #returnSet |= set(word_tokenize(re.sub('[^0-9a-zA-Z]+', ' ', word)))
#    returnSet |= set(re.sub('[^0-9a-zA-Z]+', ' ', word).split(" "))
#  return returnSet

def alphanum_only(wordSet): 
  #From Kevin Wu: use a list comprehension instead of a loop for speed
  #https://bitbucket.org/bejerano/clinphen/branch/feature/performance-improvements
  return set(itertools.chain.from_iterable([re.sub('[^0-9a-zA-Z]+', ' ', word).split(" ") for word in wordSet]))


def sort_ids_by_occurrences_then_earliness(id_to_lines):
  listForm = []
  for hpoid in id_to_lines.keys(): listForm.append((hpoid, len(id_to_lines[hpoid]), min(id_to_lines[hpoid])))
  listForm.sort(key=lambda x: [-1*x[1], x[2], x[0]])
  returnList = list()
  for item in listForm: returnList.append(item[0])
  return returnList

##extracts each HPO term that was in the text, what words matched the term, and what in what context the term was found, line by line, in the following format:
##term_id    term_name    synonym_match    earliness    positive    negative    family    healthy disease    context
def extract_phenotypes(text,
                       id_to_name_map = ID_TO_NAME_MAP,
                       id_to_syn_map = ID_TO_SYN_MAP
                       ):
  # po is too too common of a sequence. Easier to remove it from the ontology and expand it here.
  text = re.sub(r'\bpo\b', 'per os', text)
  text = re.sub(r'\bPO\b', 'per os', text)
  text = re.sub(r'\bp/o\b', 'per os', text)
  text = re.sub(r'\bp\.o\.', 'per os', text)
  tag_ID_to_lines = dict()#defaultdict(set)
  tag_ID_to_flags = defaultdict(Counter) 
  tag_ID_to_syn_match = dict() 
  medical_record = load_medical_record_subsentences(text)
  medical_record_subsentences = []
  medical_record_words = []
  medical_record_flags = []
  subsent_to_sentence = []
  ix = 0
  for subsents in medical_record:
    whole_sentence = ""
    for subsent in subsents: whole_sentence += subsent + " "
    whole_sentence = whole_sentence.strip()
    whole_sentence = re.sub('[^0-9a-zA-Z]+', ' ', whole_sentence)
    # flags = get_flags(whole_sentence.split(" "),flag_dict = FLAGS) 
    flags = []
    for subsent in subsents:
      medical_record_subsentences.append(subsent)
      subsent_to_sentence.append(whole_sentence)
      medical_record_words.append(add_lemmas(alphanum_only(set([subsent]))))
      medical_record_flags.append(flags)
  mr_map = load_mr_map(medical_record_words)
  #syns = load_all_hpo_synonyms(id_to_syn_map)
  for hpoID in id_to_syn_map.keys():
    for syn in id_to_syn_map[hpoID]:
      syn = re.sub('[^0-9a-zA-Z]+', ' ', syn.lower())
      synTokens = alphanum_only(set([syn]))
      if len(synTokens) < 1: continue
      firstToken = list(synTokens)[0]
      lines = set(mr_map[firstToken])
      for token in synTokens:
        lines &= set(mr_map[token])
        if len(lines) < 1: break
      if len(lines) < 1: continue
      for i in lines:
        line = " ".join(medical_record_words[i])
        flagged = [] 
        # for flag in medical_record_flags[i]:
        #   if flag not in synTokens:
        #     flagged.append(FLAG_TYPE[flag]) 
        if flagged == []:
          flagged.append('positive')
        tag_ID_to_lines[(hpoID,ix)] = i
        tag_ID_to_flags[(hpoID,ix)].update(flagged)
        tag_ID_to_syn_match[(hpoID,ix)] = syn
        ix += 1

  # returnString = ["term_id\tterm_name\tsynonym_match\tcontext\tearliness\t" + 'positive\t' + '\t'.join(FLAGS)]
  returnString = ["term_id\tterm_name\tsynonym_match\tcontext\tearliness\t" + 'positive']
  for ID,ix in tag_ID_to_flags: returnString.append(
    "\t".join(
      [
        ID, 
        id_to_name_map[ID], 
        tag_ID_to_syn_match[(ID,ix)], #synonym match goes here 
        subsent_to_sentence[tag_ID_to_lines[(ID,ix)]],
        str(tag_ID_to_lines[(ID,ix)])
        ] + [str(1*(tag_ID_to_flags[(ID,ix)][flag_type]>0)) for flag_type in ['positive']]
        # ] + [str(1*(tag_ID_to_flags[(ID,ix)][flag_type]>0)) for flag_type in ['positive']+list(FLAGS)]
      )
    )
  return "\n".join(returnString)

def extract_phenotypes_df(txt, id_to_name_map = ID_TO_NAME_MAP, id_to_syn_map = ID_TO_SYN_MAP):
  dat = extract_phenotypes(txt, id_to_name_map = id_to_name_map, id_to_syn_map = id_to_syn_map)
  head = dat.split('\n')[0].split('\t')
  rows = [item.split('\t') for item in dat.split('\n')[1:]]
  res = pd.DataFrame(
    data = rows,
    columns = head
    )
  # for f in ['earliness','positive'] + list(FLAGS):
  for f in ['earliness','positive']:
    res[f] = res[f].astype(int)
  return res