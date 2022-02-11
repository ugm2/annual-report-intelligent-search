import click
from neural_search.core.index import index_docs
from jina import Flow

def query(query_flow_path):
    flow = Flow.load_config(query_flow_path)
    flow.rest_api = True
    flow.protocol = 'http'
    with flow:
        flow.block()

@click.command()
@click.option('--task', '-t',
              type=click.Choice(['index', 'query'], case_sensitive=False))
@click.option('--path', '-p',
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              default='./data/books.csv')
@click.option('--num_docs', '-n',
              type=click.INT, default=None)
@click.option('--index_field', '-i',
              type=click.STRING, default='Name')
@click.option('--index_flow_path', '-f',
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              default='./flows/index.yml')
@click.option('--query_flow_path', '-f',
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              default='./flows/query.yml')
@click.option('--num_docs', '-n', default=1000)
def main(task, num_docs, path, index_field, index_flow_path, query_flow_path):
    """
    Main function for running the server.
    """
    if task == 'index':
        index_docs(
            path=path,
            num_docs=num_docs,
            index_field=index_field,
            index_flow_path=index_flow_path
        )
    elif task == 'query':
        query(query_flow_path)
    else:
        raise NotImplementedError(
            f'Unknown task: {task}.')


if __name__ == "__main__":
    main()