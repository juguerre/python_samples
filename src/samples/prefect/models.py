import importlib
from typing import Annotated, Any, Callable, Literal, Self

from prefect import Flow
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from ruamel.yaml import YAML

from samples.dag_runner import FunctionTask


class MappingTask(FunctionTask):
    model_type: Literal["mapping_task"] = "mapping_task"
    field_mapping: dict[str, Any]
    func_name: str = "samples.prefect.sample_funcs:mapping"


class ElasticExtractTask(FunctionTask):
    model_type: Literal["elastic_extract_task"] = "elastic_extract_task"
    index: str
    data_source: str
    filter: dict[str, Any]
    func_name: str = "samples.prefect.sample_funcs:elastic_extract"


class CSVExtractTask(FunctionTask):
    model_type: Literal["csv_extract_task"] = "csv_extract_task"
    data_source: str
    path: str
    func_name: str = "samples.prefect.sample_funcs:csv_extract"


class PythonExtractTask(FunctionTask):
    model_type: Literal["python_extract_task"] = "python_extract_task"
    data_source: str


class PythonTransformTask(FunctionTask):
    model_type: Literal["python_transform_task"] = "python_transform_task"


class DBLoadTask(FunctionTask):
    model_type: Literal["db_load_task"] = "db_load_task"
    data_asset: str
    table: str
    connection_id: str
    func_name: str = "samples.prefect.sample_funcs:db_load"


Task = Annotated[
    FunctionTask
    | MappingTask
    | ElasticExtractTask
    | CSVExtractTask
    | PythonExtractTask
    | PythonTransformTask
    | DBLoadTask,
    Field(discriminator="model_type"),
]


class PrefectWorkPoolConfig(BaseModel):
    name: str
    work_queue_name: str | None
    job_variables: dict[str, Any]


class PrefectDeployConfig(BaseModel):
    source: str
    deploy_name: str
    entrypoint: str
    func_name: str
    _entry_func: Callable | None = None
    work_pool_id: str | None = None
    pull_config_id: str | None = None

    @property
    def entry_func(self) -> Flow:
        if not self._entry_func:
            module, func = self.func_name.split(":")
            self._entry_func: Flow = getattr(importlib.import_module(module), func)
        return self._entry_func

    @field_validator("work_pool_id", mode="after")
    @classmethod
    def validate_work_pool_id(cls, v):
        if v is None:
            return cls.default_work_pool
        return v


class PipelineBaseConfig(BaseModel):
    prefect_deploy: PrefectDeployConfig


class ETLBaseConfig(BaseModel):
    tasks: dict[str, Task]

    @field_validator("tasks", mode="before")
    @classmethod
    def set_task_ids(cls, v):
        for k, task in v.items():
            task["task_id"] = k
        return v


class ETLExtractConfig(ETLBaseConfig):
    pass


class ETLTransformConfig(ETLBaseConfig):
    pass


class ETLLoadConfig(ETLBaseConfig):
    pass


class ETLPipelineConfig(PipelineBaseConfig):
    model_type: Literal["etl_pipeline"] = "etl_pipeline"
    prefect_deploy: PrefectDeployConfig
    extract: ETLExtractConfig
    transform: ETLTransformConfig
    load: ETLLoadConfig


class FreePipelineConfig(PipelineBaseConfig):
    model_type: Literal["free_pipeline"] = "free_pipeline"
    tasks: list[Task]


# Define the discriminated union
Pipeline = Annotated[
    ETLPipelineConfig | FreePipelineConfig, Field(discriminator="model_type")
]


class JobConfig(RootModel[dict[str, Pipeline]]):
    root: dict[str, Pipeline]


class Config(BaseModel):
    default_work_pool: str
    default_pull_config: str
    jobs: dict[str, JobConfig]

    @model_validator(mode="after")
    def validate_defaults(self):
        """set default work pool and pull config for all pipelines when none is specified"""
        for job in self.jobs.values():
            for pipeline in job.root.values():
                pipeline.prefect_deploy.work_pool_id = (
                    pipeline.prefect_deploy.work_pool_id or self.default_work_pool
                )
                pipeline.prefect_deploy.pull_config_id = (
                    pipeline.prefect_deploy.pull_config_id or self.default_pull_config
                )
        return self

    @classmethod
    def load_config_from_yaml(cls, path: str) -> Self:
        # use ruamel to load yaml safely
        yaml = YAML(typ="safe")
        with open(path, "r") as f:
            return cls(**yaml.load(f))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    config = Config.load_config_from_yaml("src/samples/prefect/component_config.yaml")
    # print config as json
    # print(config.model_dump_json(indent=2))
    config.jobs["sample_job"].root["etl1_ppl"].prefect_deploy.entry_func()

    # flow: Flow = Flow.from_source(
    #     source=config.jobs["sample_job"].root["etl1_ppl"].prefect_deploy.source,
    #     entrypoint=config.jobs["sample_job"].root["etl1_ppl"].prefect_deploy.entrypoint,
    # )

    # flow.deploy(
    #     name=config.jobs["sample_job"].root["etl1_ppl"].prefect_deploy.deploy_name,
    #     work_pool_name=config.jobs["sample_job"]
    #     .root["etl1_ppl"]
    #     .prefect_deploy.work_pool_id,
    # )
