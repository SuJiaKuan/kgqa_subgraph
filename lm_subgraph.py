import json
import os

import numpy as np

from model import SentenceEncoder


LABEL_RELATIONS = [
    "<fb:type.object.name>",
]

RELATION_LABEL_MAPPING = {
    "<fb:type.object.type>": "is",
}


def cal_cosine(x, y):
    norm_x = x / np.linalg.norm(x)
    norm_y = y / np.linalg.norm(x)

    return float(np.dot(norm_x, norm_y))


def load_facts(facts_file):
    if not os.path.exists(facts_file):
        return {}

    fact_mapping = {}

    with open(facts_file) as f:
        fact_lines = list(f)

    for fact_line in fact_lines:
        subj_entity, relation, obj_entity = fact_line.strip().split(None, 2)
        '''
        if subj_entity not in fact_mapping:
            fact_mapping[subj_entity] = {relation: obj_entity}
        else:
            fact_mapping[subj_entity][relation] = obj_entity
        '''
        if subj_entity not in fact_mapping:
            fact_mapping[subj_entity] = [(relation, obj_entity)]
        else:
            fact_mapping[subj_entity].append((relation, obj_entity))

    return fact_mapping


def get_entity_label(entity_id, entity_facts):
    if not entity_id.startswith("<fb:"):
        return entity_id

    if not entity_id.startswith("<fb:m."):
        return get_relation_label(entity_id)

    relations = [f[0] for f in entity_facts]
    for label_relation in LABEL_RELATIONS:
        if label_relation in relations:
            return entity_facts[relations.index(label_relation)][1]

    return entity_id


def get_relation_label(relation):
    if relation in RELATION_LABEL_MAPPING:
        return RELATION_LABEL_MAPPING[relation]

    phrase = relation.split(".")[-1][:-1]
    words = phrase.split("_")
    words = list(filter(lambda w: len(w) > 1, words))

    return " ".join(words)


def select_entities(
    question_text,
    sentence_encoder,
    seed_entity_ids,
    fact_mapping,
    num_hops,
    beam_size,
):
    question_vector = sentence_encoder.encode(question_text.lower())
    print(question_text)

    selected_entity_ids = set(seed_entity_ids)

    last_entity_ids = seed_entity_ids
    for _ in range(num_hops):
        candidates = []
        # print("====")
        for subj_id in last_entity_ids:
            subj_facts = fact_mapping.get(subj_id, [])
            subj_label = get_entity_label(subj_id, subj_facts)
            for relation, obj_id in subj_facts:
                obj_facts = fact_mapping.get(obj_id, [])
                obj_label = get_entity_label(obj_id, obj_facts)
                relation_label = get_relation_label(relation)
                fact_text = "{} {} {}".format(
                    subj_label,
                    relation_label,
                    obj_label,
                )
                fact_vector = sentence_encoder.encode(fact_text.lower())
                similarity = cal_cosine(question_vector, fact_vector)
                candidates.append((obj_id, fact_text, similarity))

        top_candidates = \
            sorted(candidates, key=lambda f: f[2], reverse=True)[:beam_size]
        '''
        for candidate in top_candidates:
            print(candidate)
        '''
        top_entity_ids = [c[0] for c in top_candidates]
        selected_entity_ids = selected_entity_ids.union(top_entity_ids)
        last_entity_ids = top_entity_ids

    return list(selected_entity_ids)


def main():
    target_set = "train"
    target_set = "dev"
    num_hops = 1
    num_hops = 2
    beam_size = 500

    sim_cache_filepath = "./lm_processed/sim_cache.pickle"

    subgraph_dir = "freebase_2hops/stagg.neighborhoods/"
    question_json = "scratch/webqsp_processed.json"
    range_json = "graftnet_processed/full/{}_bt.json".format(target_set)
    output_json = "./lm_processed/{}_bt.json".format(target_set)

    with open(question_json, "r") as f:
        questions = json.load(f)

    with open(range_json, "r") as f:
        question_ranges = json.load(f)
        question_ranges = [q["id"] for q in question_ranges]

    sentence_encoder = SentenceEncoder(sim_cache_filepath)

    results = []
    for question in questions[20:]:
        question_id = question["QuestionId"]
        if question_id not in question_ranges:
            continue

        question_text = question["QuestionText"]
        seed_entity_ids = [e["freebaseId"] for e in question["OracleEntities"]]
        answers = [{
            "kb_id": a["freebaseId"],
            "text": a["freebaseId"],
        } for a in question["Answers"]]

        facts_file = os.path.join(subgraph_dir, "{}.nxhd".format(question_id))
        fact_mapping = load_facts(facts_file)
        entity_ids = select_entities(
            question_text,
            sentence_encoder,
            seed_entity_ids,
            fact_mapping,
            num_hops,
            beam_size,
        )
        subgraph_entities = [{
            "kb_id": e,
            "text": e,
        } for e in entity_ids]
        print(question_id, len(seed_entity_ids), len(entity_ids))

        results.append({
            "id": question_id,
            "question": question_text,
            "subgraph": {
                "entities": subgraph_entities,
            },
            "answers": answers,
        })

    with open(output_json, "w") as f:
        f.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
