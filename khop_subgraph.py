import json
import os
import random


def read_facts(facts_file):
    if not os.path.exists(facts_file):
        return {}

    fact_mapping = {}

    with open(facts_file) as f:
        fact_lines = list(f)

    for fact_line in fact_lines:
        subj_entity, relation, obj_entity = fact_line.strip().split(None, 2)
        if subj_entity not in fact_mapping:
            fact_mapping[subj_entity] = [obj_entity]
        else:
            fact_mapping[subj_entity].append(obj_entity)

    return fact_mapping


def get_khop(seed_entity_ids, fact_mapping, num_hops, pruning_size=0):
    khop_entitity_ids = set(seed_entity_ids)

    last_entity_ids = seed_entity_ids
    cur_entity_ids = []
    for _ in range(num_hops):
        for entity_id in last_entity_ids:
            if entity_id in fact_mapping:
                cur_entity_ids += fact_mapping[entity_id]
        khop_entitity_ids = khop_entitity_ids.union(cur_entity_ids)
        if pruning_size > 0:
            num_choices = min(pruning_size, len(cur_entity_ids))
            last_entity_ids = \
                random.choices(cur_entity_ids, k=num_choices) \
                if cur_entity_ids \
                else []
        else:
            last_entity_ids = cur_entity_ids
        cur_entity_ids = []

    return list(khop_entitity_ids)


def main():
    target_set = "dev"
    target_set = "train"
    num_hops = 1
    num_hops = 2
    pruning_size = 10000

    subgraph_dir = "freebase_2hops/stagg.neighborhoods/"
    question_json = "scratch/webqsp_processed.json"
    range_json = "graftnet_processed/full/{}_bt.json".format(target_set)
    output_json = "./khop_processed/{}_bt.json".format(target_set)

    with open(question_json, "r") as f:
        questions = json.load(f)

    with open(range_json, "r") as f:
        question_ranges = json.load(f)
        question_ranges = [q["id"] for q in question_ranges]

    results = []
    for question in questions:
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
        fact_mapping = read_facts(facts_file)
        entity_ids = get_khop(
            seed_entity_ids,
            fact_mapping,
            num_hops,
            pruning_size=pruning_size,
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
