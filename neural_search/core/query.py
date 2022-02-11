from jina import Flow

class Query:

    def query(self, query_flow_path):
        flow = Flow.load_config(query_flow_path)
        flow.rest_api = True
        flow.protocol = 'http'
        with flow:
            flow.block()