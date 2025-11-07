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
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import date, timedelta, datetime
from PIL import Image
import numpy as np
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
import re
import os

client_id = "" 
client_secret = ""

config = SHConfig()
config.sh_client_id = client_id
config.sh_client_secret = client_secret
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
config.save("cdse")

config = SHConfig("cdse")

client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)

resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
print(resp.content)

def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response

oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

aoi_coords_wgs84 = [-17.954016,32.018547,-15.878085,33.601334]

resolution = 80
aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {aoi_size} pixels")

catalog = SentinelHubCatalog(config=config) 

aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=CRS.WGS84)

current_date = (f"{date.today().year}-{date.today().month}-{int(date.today().day) - 5}",f"{date.today().year}-{date.today().month}-{int(date.today().day)}")
print(f"Today is: {current_date[1]}")

evalscript_sentinel3_olci_IWV = """
//VERSION=3

let rangeMin = 0; 
let rangeMax = 70;  
let viz = ColorRampVisualizer.createOceanColor(rangeMin, rangeMax);

function setup() {
  return {
    input: [{
      bands: ["B18", "B19", "dataMask"]
    }],
    output: { bands: 4 }
  }
}

function evaluatePixel(samples) {
  if (samples.dataMask === 0) {
    return [0, 0, 0, samples.dataMask];
  }
  
  let B18 = samples.B18;
  let B19 = samples.B19;
  let C1 = 0.0746699;
  let C2 = -1.15649;
  let C3 = 19.9892;
  
  let value = (2 * C1 - Math.log(B19) * C2 +  Math.log(B19) * Math.log(B19) * C3 ) / 10;

  let color = viz.process(value);

  return [...color, samples.dataMask];
}
"""

evalscript_sentinel3_olci_CHL = """
//VERSION=3

let rangeMin = 0; 
let rangeMax = 1;  
let viz = ColorRampVisualizer.createOceanColor(rangeMin, rangeMax);

function setup() {
  return {
    input: [{
      bands: ["B03", "B04", "B05", "B06", "dataMask"]
    }],
    output: { bands: 4 }
  }
}

function evaluatePixel(samples) {
  if (samples.dataMask === 0) {
    return [0, 0, 0, samples.dataMask];
  }
  
  let B3 = samples.B03;
  let B4 = samples.B04;
  let B5 = samples.B05;
  let B6 = samples.B06;
  let A0 = 0.450;
  let A1 = -3.259;
  let A2 = 3.522; 
  let A3 = -3.359;
  let A4 =  0.949; 
  let R = Math.log10(Math.max(B3/B6, B4/B6, B5/B6))
  
  let chlA = Math.pow(10, A0 + A1*R + A2*R*R + A3*R*R*R + A4*R*R*R*R) 

  let color = viz.process(chlA);

  return [...color, samples.dataMask];
}
"""

evalscript_sentinel3_olci_TSM = """
//VERSION=3

let rangeMin = 0;
let rangeMax = 80;  
let viz = ColorRampVisualizer.createOceanColor(rangeMin, rangeMax);

function setup() {
  return {
    input: [{
      bands: ["B08", "B06", "dataMask"]
    }],
    output: { bands: 4 }
  }
}

function evaluatePixel(samples) {
  if (samples.dataMask === 0) {
    return [0, 0, 0, samples.dataMask];
  }
  
  let B6 = samples.B06;
  let B8 = samples.B08;

  let tsm = 190.37 * Math.pow(B8/B6,2) - 138.61 * B8/B6 + 26.883

  let color = viz.process(tsm);

  return [...color, samples.dataMask];
}
"""

evalscript_sentinel3_olci_AAE = """
//VERSION=3

let rangeMin = 0.8;  
let rangeMax = 3;   
let viz = ColorRampVisualizer.createOceanColor(rangeMin, rangeMax);

function setup() {
  return {
    input: [{
      bands: ["B06", "B17", "dataMask"]
    }],
    output: { bands: 4 }
  }
}

function evaluatePixel(samples) {
  if (samples.dataMask === 0) {
    return [0, 0, 0, samples.dataMask];
  }
  
  let AOD_B01 = samples.B06;  
  let AOD_B02 = samples.B17;  

  let aae = -Math.log(AOD_B01 / AOD_B02) / Math.log(510 / 865);

  let color = viz.process(aae);

  return [...color, samples.dataMask];
}
"""

evalscript_sentinel3_olci_OTCI = """
//VERSION=3 
const map = [ 
	[0.0, 0x00007d],
	[1.0, 0x004ccc],
	[1.8, 0xff3333],
	[2.5, 0xffe500],
	[4.0, 0x00cc19],
	[4.5, 0x00cc19],
	[5.0,0xffffff]
];

const visualizer = new ColorRampVisualizer(map);
function setup() {
	return {
		input: [ "B10", "B11", "B12", "dataMask" ],
        output: [
		{ id: "default", bands: 4 },
		{ id: "index", bands: 1, sampleType: "FLOAT32" },
        { id: "eobrowserStats", bands: 1 },
        { id: "dataMask", bands: 1},
    	]
	};
}
    
function evaluatePixel(samples) {
    let OTCI = (samples.B12- samples.B11)/(samples.B11- samples.B10);
    let imgVals = null;
    const indexVal = samples.dataMask === 1 && OTCI >= -10 && OTCI <= 10 ? OTCI : NaN;
    imgVals = [...visualizer.process(OTCI), samples.dataMask]
    return {
        default: imgVals,
        index: [indexVal],
        eobrowserStats:[indexVal],
        dataMask: [samples.dataMask]      
    };
 }
"""

evalscript_sentinel3_olci_RGB = """
//VERSION=3 (auto-converted from 1)
let minVal = 0.0;
let maxVal = 0.8;

let viz = new HighlightCompressVisualizer(minVal, maxVal);

function evaluatePixel(samples) {
    let val = [samples.B17, samples.B05, samples.B02, samples.dataMask];
    return viz.processList(val);
}

function setup() {
  return {
    input: [{
      bands: ["B17", "B05", "B02" , "dataMask" ]
    }],
    output: { bands: 4 }
  }
}
"""

def save_image_as_jpeg(image_array, filename):
    output_dir = "images/copernicusS3"
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, filename)
    
    image = Image.fromarray(np.uint8(image_array))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(full_path, "JPEG")
    print(f"Image saved as {full_path}")

def request_sentinel(data, image_name, change_date=1, start_date="2022-05-01", end_date="2022-05-20", save=0):
    time_interval = (start_date, end_date) if change_date == 0 else current_date

    treated_data = SentinelHubRequest(
        evalscript=data,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL3_OLCI.define_from(
                    name="s3olci", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=time_interval,
                other_args={"dataFilter": {"mosaickingOrder": "mostRecent"}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,  
        size=aoi_size,  
        config=config,  
    )

    final_data = treated_data.get_data()
    if save == 0:
      save_image_as_jpeg(final_data[0], image_name)
    else: 
       return final_data[0]

vals = [evalscript_sentinel3_olci_IWV, evalscript_sentinel3_olci_CHL, evalscript_sentinel3_olci_TSM, evalscript_sentinel3_olci_AAE, 
        evalscript_sentinel3_olci_OTCI, evalscript_sentinel3_olci_RGB]
image_names = ["IWV.jpeg", "CHL.jpeg", "TSM.jpeg", "AAE.jpeg", "OTCI.jpeg", "RGB.jpeg"]
excluded = ["OTCI.jpeg", "RGB.jpeg"]
units = [f"kg m\u2212\u00B2", f"µg L\u207B\u00B9", f"mg L\u207B\u00B9","N/A", "N/A", "N/A"]

def apply_transparency(reference_image_path, original_image_path, output_image_path):
    output_dir = "images/copernicusS3"
    os.makedirs(output_dir, exist_ok=True)
    
    reference_full_path = os.path.join(output_dir, reference_image_path)
    original_full_path = os.path.join(output_dir, original_image_path)
    output_full_path = os.path.join(output_dir, output_image_path)
    
    reference_image = Image.open(reference_full_path)
    original_image = Image.open(original_full_path).convert("RGBA")

    ref_array = np.array(reference_image)
    orig_array = np.array(original_image)

    rgb_sum = np.sum(ref_array[:, :, :3], axis=-1)
    sum_mask = rgb_sum >= 180
    
    specific_rgb_mask = (ref_array[:, :, 0] < 50) & (ref_array[:, :, 1] < 50) & (ref_array[:, :, 2] > 100)
    combined_mask = sum_mask | specific_rgb_mask

    orig_array[sum_mask] = [255, 255, 255, 255]
    result_image = Image.fromarray(orig_array, "RGBA")
    
    result_image.save(output_full_path, "PNG")
    print(f"Image with cloud masked saved as {output_full_path}")

def cloud_mask():
  for name in image_names:
      if name in excluded:
          continue
      else:
          apply_transparency('RGB.jpeg', name, name )

def land_filter(filename, units="N/A"):
  output_dir = "images/copernicusS3"
  os.makedirs(output_dir, exist_ok=True)
  
  full_path = os.path.join(output_dir, filename)
  
  data_crs = ccrs.PlateCarree()
  coast = cfeature.GSHHSFeature(scale='full')

  fig = plt.figure(figsize=(12, 9))
  ax = fig.add_subplot(projection=data_crs)

  ax.add_feature(cfeature.LAND, zorder=10)
  ax.add_feature(coast, linewidth=1.2, zorder=10)

  ax.set_extent([aoi_coords_wgs84[0], aoi_coords_wgs84[2], aoi_coords_wgs84[1], aoi_coords_wgs84[3]], crs=data_crs) 

  image_path = full_path
  image = Image.open(image_path).convert("RGBA")

  image_extent = [aoi_coords_wgs84[0], aoi_coords_wgs84[2], aoi_coords_wgs84[1], aoi_coords_wgs84[3]]

  ax.imshow(image, origin='upper', extent=image_extent, transform=data_crs, zorder=2)

  gl = ax.gridlines(draw_labels=True, color="None", xlocs=np.arange(-20, -14, 1), ylocs=np.arange(31, 36, 1))
  gl.top_labels = False
  gl.right_labels = False
  gl.ylabel_style = {'rotation': 90}

  savename = filename.split(".")[0]
  plt.title(f"{savename} {units} @ " + str(current_date[1]))

  print(f"Image with land masked saved as {full_path}")
  plt.savefig(full_path, dpi=300, bbox_inches='tight', format='jpeg')

x_coords = np.linspace(aoi_coords_wgs84[0], aoi_coords_wgs84[2], aoi_size[0]) 
y_coords = np.linspace(aoi_coords_wgs84[1], aoi_coords_wgs84[3], aoi_size[1]) 
x, y = np.meshgrid(x_coords, y_coords)

def land_mask():
  cnt = 0
  for name in image_names:
      if name in excluded:
          continue
      else:
          land_filter(name, units[cnt])
      cnt += 1

def create_ocean_colormap_image(rangeMin, rangeMax, width, height, output_filename, style='jet'):
    output_dir = "images/copernicusS3"
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, output_filename)
    
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    gradient = np.linspace(rangeMin, rangeMax, 100).reshape(1, -1)
    gradient = np.tile(gradient, (height, 1))

    cax = ax.imshow(gradient, aspect='auto', cmap=style, extent=[0, width, rangeMin, rangeMax])

    ax.axis('off')

    cbar = fig.colorbar(cax, orientation='horizontal', ax=ax, fraction=1.0, pad=0.0, extend='both')
    cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.savefig(full_path, format='jpeg', bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.close(fig)

    print(f"Colorbar image saved as {full_path}")

def image_merge(image_path, legend_path):
    output_dir = "images/copernicusS3"
    os.makedirs(output_dir, exist_ok=True)
    
    image_full_path = os.path.join(output_dir, image_path)
    legend_full_path = os.path.join(output_dir, legend_path)
    
    image1 = Image.open(image_full_path)
    image2 = Image.open(legend_full_path)

    image1_size = image1.size
    image2_size = image2.size

    new_width = max(image1_size[0], image2_size[0])
    new_height = image1_size[1] + image2_size[1]
    new_image = Image.new("RGB", (new_width, new_height), (250, 250, 250, 255))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, image1_size[1]))

    new_image.save(image_full_path)

def extract_values(input_string):
    pattern = r'let\s+rangeMin\s*=\s*(\d+\.?\d*);.*?let\s+rangeMax\s*=\s*(\d+\.?\d*);'
    
    match = re.search(pattern, input_string, re.DOTALL)
    
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        return min_val, max_val
    else:
        raise ValueError("Values not found in the input string")

def legends():
   cnt = 0
   for element in image_names:
      if element in excluded:
         continue
      else:  
        min_val, max_val = extract_values(vals[cnt])
        create_ocean_colormap_image(min_val, max_val, 950, 100, 'colorbar_only.jpeg')
        image_merge(element, "colorbar_only.jpeg")
      cnt += 1
        
def date_chooser():
    time_interval = date.today() - timedelta(days = 3), date.today()
    search_iterator = catalog.search(
        DataCollection.SENTINEL3_OLCI,
        bbox=aoi_bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime"], "exclude": []},
    )

    results = list(search_iterator)
    tile_to_find = "_NT_"
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
    
def available_data():
    time_interval = date.today() - timedelta(days = 3), date.today()
    search_iterator = catalog.search(
        DataCollection.SENTINEL3_OLCI,
        bbox=aoi_bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime"], "exclude": []},
    )

    results = list(search_iterator)
    for element in results:
       print(element)
   
current_date = date_chooser()
print(f"Using: {current_date}")

def daily_images(update = True, cloud_filter= True, map = True):
  if update:
    cnt = 0
    for element in vals:
        request_sentinel(element, image_names[cnt],1,"2024-09-01", "2024-09-03")
        cnt+=1
  if cloud_filter:
    cloud_mask()
  if map:
     land_mask()
     legends()
     
available_data()
daily_images(True,True,True)