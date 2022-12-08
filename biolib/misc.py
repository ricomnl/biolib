import datetime
import time
import uuid
from pathlib import Path

from google.cloud import storage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from kedro.extras.datasets.matplotlib import MatplotlibWriter


TWO_FIGS_RIGHT = {
    'scaleX': 1/2,
    'scaleY': 1/2,
    'translateX': 4560000,
    'translateY': 1300000,
    'unit': 'EMU',
}
TWO_FIGS_LEFT = {
    'scaleX': 1/2,
    'scaleY': 1/2,
    'translateX': 0,
    'translateY': 1300000,
    'unit': 'EMU',
}
TWO_FIGS_TOP = {
    'scaleX': 1/2,
    'scaleY': 1/2,
    'translateX': 0,
    'translateY': 0,
    'unit': 'EMU',
}
TWO_FIGS_BOTTOM = {
    'scaleX': 1/2,
    'scaleY': 1/2,
    'translateX': 0,
    'translateY': 1300000,
    'unit': 'EMU',
}


class Timer:
    """Timer class for timing code blocks.
    """
    def __enter__(self):
        self.tick = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.tock = time.time()
        self.elapsed = self.tock - self.tick


def generate_download_signed_url_v4(bucket_name, blob_name, creds, minutes=15, method='GET'):
    """Generates a v4 signed URL for downloading a blob.

    Note that this method requires a service account key file. You can not use
    this if you are using Application Default Credentials from Google Compute
    Engine or from the Google Cloud SDK.

    Parameters
    ----------
    bucket_name : str
        Name of the bucket to access.
    blob_name : str
        Name of the blob to access.
    creds : google.auth.credentials.Credentials
        Credentials to use for the signed URL.
    minutes : int, optional
        Number of minutes the signed URL should be valid for. Default: 15.
    method : str, optional
        HTTP method to use for the signed URL. Default: 'GET'.
    
    Returns
    -------
    str
        The signed URL.    
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        credentials=creds,
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=minutes),
        # Allow GET requests using this URL.
        method=method,
    )
    return url


def create_image(presentation_id, page_id, url, creds, size_dict={}, transform_dict={}):
    """Creates an image in a Google Slides presentation.
    
    Parameters
    ----------
    presentation_id : str
        The ID of the presentation to create the image in.
    page_id : str
        The ID of the page to create the image in.
    url : str
        The URL of the image to create.
    creds : google.auth.credentials.Credentials
        Credentials to use for the signed URL.
    size_dict : dict, optional
        Dictionary containing the size of the image, by default {}
        Example: {'height': {'magnitude': 4000000, 'unit': 'EMU'}, 'width': {'magnitude': 4000000, 'unit': 'EMU'}}
    transform_dict : dict, optional
        Dictionary containing the transform of the image, by default {}
        Example: {'scaleX': 1, 'scaleY': 1, 'translateX': 100000, 'translateY': 100000, 'unit': 'EMU'}
    """
    try:
        service = build('slides', 'v1', credentials=creds)
        requests = []
        image_id = f'image_{str(uuid.uuid4())}'
        requests.append({
            'createImage': {
                'objectId': image_id,
                'url': url,
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': size_dict,
                    'transform': transform_dict,
                }
            }
        })

        body = {
            'requests': requests
        }
        response = service.presentations() \
            .batchUpdate(presentationId=presentation_id, body=body).execute()
        create_image_response = response.get('replies')[0].get('createImage')
        print(f"Created image with ID: {(create_image_response.get('objectId'))}")

        return response
    except HttpError as error:
        print(f"An error occurred: {error}")
        print("Images not created")
        return error


def add_plt_to_slides(
    plot, 
    output_path, 
    bucket_name, 
    blob_name, 
    presentation, 
    creds,
    new_slide=True,
    slide_title=' ',
    slide_notes=' ',
    slide_layout=(1,1),
):
    """Appends a matplotlib plot to a Google Slides presentation.

    Parameters
    ----------  
    plot : matplotlib.pyplot
        The plot to append.
    output_path : str or pathlib.Path
        The path to save the plot to.
    bucket_name : str
        The name of the bucket to save the plot to.
    blob_name : str
        The name of the blob to save the plot to.
    presentation : gslides.Presentation
        The presentation to append the plot to.
    creds : google.auth.credentials.Credentials
        Credentials to use for the signed URL.
    new_slide : bool, optional
        Whether to create a new slide or append to an existing one. Default: True.
    slide_title : str, optional
        The title of the slide to append the plot to. Default: ' '.
    slide_notes : str, optional
        The notes of the slide to append the plot to. Default: ' '.
    slide_layout : tuple, optional
        The layout of the slide to append the plot to. Default: (1,1).
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)
    single_plot_writer = MatplotlibWriter(
        filepath=output_path.as_posix()
    )
    single_plot_writer.save(plot)
    url = generate_download_signed_url_v4(
        bucket_name=bucket_name, blob_name=blob_name, creds=creds
    )
    if new_slide:
        presentation.add_slide(objects=[], layout=slide_layout, title=slide_title, notes=slide_notes)
    create_image(presentation.pr_id, page_id=presentation.slide_ids[-1], creds=creds, url=url)