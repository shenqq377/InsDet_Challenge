import random
import pickle
import numpy as np
import json
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval


def get_precisions(gt_file, result_file, metric="bbox"):
    """plot precision-recall curve based on testing results of json file.
    Args:
        gt_file: json file of ground-truth.
        result_file: json file of testing results.
        iou: list [0.5:0.05:0.95]
        metric: Metrics to be evaluated. Options are 'bbox', 'segm'.
    """
    # load ground-truth
    coco_gt = COCO(annotation_file=gt_file)

    # load testing results
    if '.json' in result_file:
        with open(result_file) as f:
            json_results = json.load(f)
        coco_dt = coco_gt.loadRes(json_results)

        # initialize COCOeval instance
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # extract eval data
        res = coco_eval.stats 
    else:
        raise RuntimeError("only json file submission is supported!")
    
    return res

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    print(kwargs["submission_metadata"])
    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        res = get_precisions(test_annotation_file, user_submission_file) # all 160 imgs
        # easy scenes
        # hard scenes
        output["result"] = [
            {
                "val_split":{
                    "AP": res[0],
                    "AP50": res[1],
                    "AP75": res[2],
                    "AP_easy": res[1],
                    "AP_hard": res[1],
                    "AP_small": res[3],
                    "AP_medium": res[4],
                    "AP_large": res[5],
                    "AR": res[1],
                }
            }
        ]

        # To display the results in the result file
        output["submission_result"] = output["result"][0]["val_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        res = get_precisions(test_annotation_file, user_submission_file) # all 300+ imgs
        # easy scenes
        # hard scenes
        output["result"] = [
            {
                "test_split":{
                    "AP": res[0],
                    "AP50": res[1],
                    "AP75": res[2],
                    "AP_easy": res[1],
                    "AP_hard": res[1],
                    "AP_small": res[3],
                    "AP_medium": res[4],
                    "AP_large": res[5],
                    "AR": res[1],
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")
    return output
