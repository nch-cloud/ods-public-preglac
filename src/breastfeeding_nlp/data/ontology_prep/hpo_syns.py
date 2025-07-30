#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
from collections import defaultdict

# Input HPO file provided as a command-line argument
HPO_FILE = sys.argv[1]

# Root ID for all phenotypic abnormalities in the HPO ontology
PHENOTYPIC_ABNORMALITY_ID = "FPO:0000001"

def get_subdag(dag, key):
    """
    Given an HPO DAG and a key, returns the sub-DAG rooted at the given key.
    A DAG is represented as a map from parent to child nodes.

    :param dag: Dictionary representing the parent-to-children relationships in the DAG
    :param key: The root node for which the sub-DAG is to be computed
    :return: A set containing all nodes in the sub-DAG
    """
    return_set = set()
    return_set.add(key)  # Add current node to set
    for child in dag[key]:  # Recursively add all descendants
        return_set = return_set | get_subdag(dag, child)
    return return_set


def get_hpo_dag():
    """
    Constructs the HPO DAG (parent-to-children relationships) from the input HPO file.

    :return: A dictionary mapping each parent node to its list of child nodes
    """
    parent_to_children = defaultdict(list)

    with open(HPO_FILE) as hpo:
        for line in hpo:
            if "[Term]" in line:  # Start of new term
                hpo_id = ""  # Reset ID for new term
                continue
            line_data = line.strip().split(": ")
            if len(line_data) < 2:  # Skip lines without key-value pairs
                continue
            key, value = line_data[0], ": ".join(line_data[1:])  # Handle colons in value
            if key == "id":  # Store current term's ID
                hpo_id = value
            if key == "is_a":  # Add parent-child relationship
                parent_to_children[value.split(" ")[0]].append(hpo_id)

    return parent_to_children


def load_hpo_dag_bilateral():
    """
    Constructs both parent-to-children and child-to-parents relationships from the HPO file.

    :return: Two dictionaries, one for parent-to-children and another for child-to-parents relationships
    """
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)

    with open(HPO_FILE) as hpo:
        for line in hpo:
            if "[Term]" in line:  # Start of new term
                hpo_id = ""  # Reset ID for new term
                continue
            line_data = line.strip().split(": ")
            if len(line_data) < 2:  # Skip lines without key-value pairs
                continue
            key, value = line_data[0], ": ".join(line_data[1:])  # Handle colons in value
            if key == "id":  # Store current term's ID
                hpo_id = value
            if key == "is_a":  # Add bidirectional relationships
                parent = value.split(" ")[0]  # Extract parent ID
                parent_to_children[parent].append(hpo_id)
                child_to_parents[hpo_id].append(parent)

    return parent_to_children, child_to_parents


def get_phenotypic_abnormalities():
    """
    Builds and returns the Phenotypic Abnormality sub-DAG, starting from the root ID.

    :return: A set of all HPO IDs representing phenotypic abnormalities
    """
    parent_to_children = get_hpo_dag()
    return get_subdag(parent_to_children, PHENOTYPIC_ABNORMALITY_ID)


def load_hpo_synonyms():
    """
    Loads HPO IDs and their associated synonymous names (case-insensitive).

    :return: A dictionary mapping HPO IDs to sets of synonymous names
    """
    # Get set of all phenotypic abnormality terms
    phenotypic_abnormalities = get_phenotypic_abnormalities()
    id_to_names = defaultdict(set)

    with open(HPO_FILE) as hpo:
        hpo_id = ""
        for line in hpo:
            if "[Term]" in line:  # Start of new term
                hpo_id = ""  # Reset ID for new term
                continue
            line_data = line.strip().split(": ")
            if len(line_data) < 2:  # Skip lines without key-value pairs
                continue
            key, value = line_data[0], ": ".join(line_data[1:])  # Handle colons in value
            if key == "id":  # Store current term's ID
                hpo_id = value
            if hpo_id not in phenotypic_abnormalities:  # Skip non-phenotype terms
                continue
            if key == "name":  # Add primary term name
                id_to_names[hpo_id].add(value.lower())
            if key == "synonym":  # Add synonym, extracting from quotes
                id_to_names[hpo_id].add(value.lower().split("\"")[1])

    return id_to_names


# Load the synonyms map for HPO IDs
syn_map = load_hpo_synonyms()

# Print each HPO ID and its synonyms in tab-separated format
for hpo in syn_map:
    for syn in syn_map[hpo]:
        print(f"{hpo}\t{syn}")