from src.CancerClassification.config.configuration import ConfigurationManager
from src.CancerClassification.components.model_evaluation_with_mlflow import Evaluation
from src.CancerClassification import logger 


STAGE_NAME = "Model Evaluation"


class EvaluationPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        config = self.config.get_evaluation_config()
        evaluation = Evaluation(config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME} Pipeline")
        pipeline = EvaluationPipeline()
        pipeline.main()
        logger.info(f"Completed {STAGE_NAME} Pipeline")
    except Exception as e:
        logger.exception(e)
        raise e
