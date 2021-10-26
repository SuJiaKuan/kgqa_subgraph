import json

import numpy as np


def get_answer_coverage(
    qa_set,
    document_mapping,
    test_document=True,
    use_document_text=False,
):
    answers = qa_set["answers"]

    subgraph_entities = qa_set["subgraph"]["entities"]
    subgraph_entities = [e["text"] for e in subgraph_entities]
    subgraph_size = len(subgraph_entities)

    if not answers:
        return 0.0, 0.0, 0.0, subgraph_size

    if test_document:
        documents = [
            document_mapping[d["document_id"]]
            for d in qa_set["passages"]
            if d["document_id"] in document_mapping
        ]
        document_entities = []
        for document in documents:
            document_entities += [
                d["text"]
                for d in document["document"]["entities"]
            ]
        document_texts = [d["document"]["text"] for d in documents]

    subgraph_found = 0
    document_found = 0
    hybrid_found = 0
    for answer in answers:
        answer_id = answer["kb_id"]
        subgraph_test = answer_id in subgraph_entities
        if subgraph_test:
            subgraph_found += 1

        if test_document:
            answer_text = \
                answer_id if answer["text"] is None else answer["text"]
            document_test = \
                any([answer_text.lower() in t.lower() for t in document_texts]) \
                if use_document_text \
                else answer_id in document_entities
            if document_test:
                document_found += 1
            if subgraph_test or document_test:
                hybrid_found += 1

    return (
        subgraph_found / len(answers),
        document_found / len(answers),
        hybrid_found / len(answers),
        subgraph_size,
    )

    '''
    entity_ids = [e["text"] for e in entities]

    for answer in answers:
        answer_id = answer["kb_id"]
        if answer_id in entity_ids:
            return 1.0

    return 0.0
    '''


def main():
    '''
    data_path = "./graftnet_processed/full/dev_bt.json"
    documents_path = "./graftnet_processed/full/documents_bt.json"

    with open(data_path, "r") as f:
        qa_sets = json.loads(f.read())

    with open(documents_path, "r") as f:
        documents = json.loads(f.read())
    document_mapping = {d["documentId"]: d for d in documents}

    for qa_set in qa_sets[:1]:
        parse_qa_set(qa_set, document_mapping)
    '''

    data_path = "./graftnet_processed/full/test_bt.json"
    data_path = "./graftnet_processed/full/dev_bt.json"
    data_path = "./graftnet_processed/kb_05/test_bt.json"
    data_path = "./graftnet_processed/kb_05/train_bt.json"
    data_path = "./graftnet_processed/kb_03/train_bt.json"
    data_path = "./graftnet_processed/kb_01/train_bt.json"
    data_path = "./graftnet_processed/full/train_bt.json"
    data_path = "./khop_processed/train_1hop.json"
    data_path = "./khop_processed/train_2hop_prune100.json"
    data_path = "./khop_processed/train_2hop_prune1000.json"
    data_path = "./lm_processed/dev_bt.json"
    data_path = "./khop_processed/dev_bt.json"

    documents_path = "./graftnet_processed/full/documents_bt.json"

    print("Input data: {}".format(data_path))

    with open(data_path, "r") as f:
        qa_sets = json.loads(f.read())

    with open(documents_path, "r") as f:
        documents = json.loads(f.read())
    document_mapping = {d["documentId"]: d for d in documents}

    subgraph_recall = 0.0
    document_recall = 0.0
    hybrid_recall = 0.0
    subgraph_sizes = []
    for qa_set in qa_sets:
        results = get_answer_coverage(
            qa_set,
            document_mapping,
            test_document=False,
            use_document_text=False,
        )
        subgraph_recall += results[0]
        document_recall += results[1]
        hybrid_recall += results[2]
        subgraph_sizes.append(results[3])

    subgraph_recall /= len(qa_sets)
    document_recall /= len(qa_sets)
    hybrid_recall /= len(qa_sets)

    print("Number of questions = {}".format(len(qa_sets)))
    print("Recall (subgraph) = {}".format(subgraph_recall))
    print("Recall (document) = {}".format(document_recall))
    print("Recall (hybrid) = {}".format(hybrid_recall))
    print("Subgraph size mean = {}".format(np.mean(subgraph_sizes)))
    print("Subgraph size std = {}".format(np.std(subgraph_sizes)))


if __name__ == "__main__":
    main()
