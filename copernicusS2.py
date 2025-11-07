import subprocess
import sys
import os

packages = ["matplotlib", "pandas", "getpass", "sentinelhub"]

def check_and_install_package():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'])
    except subprocess.CalledProcessError:
        print("pip is not installed. Please install pip and try again.")
        sys.exit(1)
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} is not installed. Installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        else:
            print(f"{package} is already installed.")

import matplotlib.pyplot as plt
import pandas as pd
import getpass
import requests
from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    SentinelHubStatistical,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
    Geometry,
)
from datetime import date
from PIL import Image
import numpy as np
import datetime
from datetime import timedelta, datetime

client_id = "" 
client_secret = ""

def get_access_token(client_id, client_secret):
    token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        access_token = token_data['access_token']
        print("Token obtained successfully")
        return access_token
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")
        return None

access_token = get_access_token(client_id, client_secret)

if access_token:
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token = access_token
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    config.save("cdse")
    
    config = SHConfig("cdse")
    
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {access_token}'})
    
    resp = session.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
    print("API connection test:", resp.status_code)

aoi_coords_wgs84 = [-17.567139,32.296420,-16.040039,33.312168]

resolution = 60
aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {aoi_size} pixels")

catalog = SentinelHubCatalog(config=config) 

aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

evalscript_SWIR = """
//VERSION=3
let minVal = 0.0;
let maxVal = 0.4;

let viz = new HighlightCompressVisualizer(minVal, maxVal);

function setup() {
  return {
    input: ["B12", "B8A", "B04","dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(samples) {
    let val = [samples.B12, samples.B8A, samples.B04,samples.dataMask];
    return viz.processList(val);
}
"""

evalscript_NDWI = """
//VERSION=3
const colorRamp1 = [
  	[0, 0xFFFFFF],
  	[1, 0x008000]
  ];
const colorRamp2 = [
  	[0, 0xFFFFFF],
  	[1, 0x0000CC]
  ];

let viz1 = new ColorRampVisualizer(colorRamp1);
let viz2 = new ColorRampVisualizer(colorRamp2);

function setup() {
  return {
    input: ["B03", "B08", "SCL","dataMask"],
    output: [
		{ id:"default", bands: 4 },
        { id: "index", bands: 1, sampleType: "FLOAT32" },
        { id: "eobrowserStats", bands: 2, sampleType: 'FLOAT32' },
        { id: "dataMask", bands: 1 }
	]
  };
}

function evaluatePixel(samples) {
  let val = index(samples.B03, samples.B08);
  let imgVals = null;
  const indexVal = samples.dataMask === 1 ? val : NaN;
  
  if (val < -0) {
    imgVals = [...viz1.process(-val), samples.dataMask];
  } else {
    imgVals = [...viz2.process(Math.sqrt(Math.sqrt(val))), samples.dataMask];
  }
  return {
    default: imgVals,
    index: [indexVal],
    eobrowserStats:[val,isCloud(samples.SCL)?1:0],
    dataMask: [samples.dataMask]
  };
}

function isCloud(scl) {
  if (scl == 3) {
    return false;
  } else if (scl == 9) {
    return true;
  } else if (scl == 8) {
    return true;
  } else if (scl == 7) {
    return false;
  } else if (scl == 10) {
    return true;
  } else if (scl == 11) {
    return false;
  } else if (scl == 1) {
    return false;
  } else if (scl == 2) {
    return false;
  }
  return false;
}
"""

current_date = (f"{date.today().year}-{date.today().month}-{int(date.today().day) - 5}",f"{date.today().year}-{date.today().month}-{int(date.today().day)}")
mode = "mostRecent"

def date_chooser():
    time_interval = date.today() - timedelta(days = 10), date.today()
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=aoi_bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime"], "exclude": []},
    )

    results = list(search_iterator)
    tile_to_find = "_R023_"
    desired_date = None

    for result in results:
        if tile_to_find in result['id']:
            desired_date = result['properties']['datetime'][:10]
            break

    if desired_date:
        previous_date = datetime.strptime(desired_date, "%Y-%m-%d") - timedelta(days=1)
        formatted_date = previous_date.strftime('%Y-%m-%d')
        return (formatted_date, desired_date)
    else:
        raise ValueError("No adequate data found in chosen period")

def save_image_as_jpeg(image_array, filename):
    output_dir = "images/copernicusS2"
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, filename)
    
    image = Image.fromarray(np.uint8(image_array))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(full_path, "JPEG")
    print(f"Image saved as {full_path}")

def request_sentinel(data, image_name, change_date=1, start_date="2022-05-01", end_date="2022-05-20"):
    time_interval = (start_date, end_date) if change_date == 0 else current_date

    treated_data = SentinelHubRequest(
        evalscript=data,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s3olci", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=time_interval,
                other_args={"dataFilter": {"mosaickingOrder": mode}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,  
        size=aoi_size,  
        config=config,  
    )

    final_data = treated_data.get_data()
    save_image_as_jpeg(final_data[0], image_name)

def available_data():
  time_interval = date.today() - timedelta(days = 6), date.today()
  search_iterator = catalog.search(
      DataCollection.SENTINEL2_L2A,
      bbox=aoi_bbox,
      time=time_interval,
      fields={"include": ["id", "properties.datetime"], "exclude": []},
  )

  results = list(search_iterator)
  for element in results:
      print(element)

vals = [evalscript_true_color, evalscript_SWIR, evalscript_NDWI]
image_names = ["TRUE_COL2.jpeg", "SWIR2.jpeg", "NDWI2.jpeg"]

cnt = 0
for element in vals:
    request_sentinel(element, image_names[cnt],0, date_chooser()[0], date_chooser()[1])
    cnt+=1

available_data()
print(f"Used date: {date_chooser()[1]}")