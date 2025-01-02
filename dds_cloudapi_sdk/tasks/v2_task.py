from dds_cloudapi_sdk.tasks.base import BaseTask


class V2Task(BaseTask):
    _api_path = None
    _api_body = None
    result = None


    def __init__(self, api_path: str, api_body: dict=None):
        super().__init__()
        self._api_path = api_path
        self._api_body = api_body

    @property
    def api_path(self):
        return self._api_path

    @property
    def api_body(self):
        return self._api_body or {}

    def format_result(self, result: dict):
        return result

    @property
    def result(self):
        return self._result

    @property
    def api_trigger_url(self):
        return f"https://{self.config.endpoint}{self.api_path}"

    @property
    def api_check_url(self):
        return f"https://{self.config.endpoint}/v2/task_status/{self.task_uuid}"
