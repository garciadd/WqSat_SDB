import numpy as np
import rasterio
from sklearn.metrics import r2_score

def scores(y_true, y_pred):
    
    # R2
    r2 = r2_score(y_true, y_pred)
    
    # MAE & RMSE
    diff = y_pred - y_true
    MAE = np.nanmean(np.abs(diff))
    MedAE = np.nanmedian(np.abs(diff))
    RMSE = np.sqrt(np.nanmean(diff**2))
    
    # Mean Bias Error (MBE)
    MBE = np.nanmean(diff)
    
    # MAPE
    d = (y_true - y_pred)/y_true
    MAPE = np.nanmean(np.abs(d))
    
    return np.round([r2, MAE, MedAE, RMSE, MBE, MAPE], 3)

def std_outliers(arr):
    low_limit = np.nanmean(arr) - 3*np.nanstd(arr)
    upp_limit = np.nanmean(arr) + 3*np.nanstd(arr)
    
    return low_limit, upp_limit

def iq_outliers(arr):
    Q1, Q3 = np.nanpercentile(arr, [25, 75])
    IQR = Q3 - Q1  #IQR is interquartile range.
    low_limit = Q1 -(1.5 * IQR)
    upp_limit = Q3 + (1.5 * IQR)
    return low_limit, upp_limit

def zscores(arr, method=1):
    if method==1:
        Zsc = (arr - np.nanmean(arr)) / np.nanstd(arr)
    elif method==2:
        Zsc = (arr - np.nanmean(arr)) / (np.nanstd(arr) / np.count_nonzero(~np.isnan(arr)))
    return Zsc

def get_RGBimage(Rband, Gband, Bband):
    
    if not (Rband.shape == Gband.shape) or not (Rband.shape == Bband.shape):
        raise ValueError('Invalid RGB Bands. The bands are not the same size!!')
    
    Rband = np.clip(Rband, np.percentile(Rband, 2.5), np.percentile(Rband, 97.5))
    Gband = np.clip(Gband, np.percentile(Gband, 2.5), np.percentile(Gband, 97.5))
    Bband = np.clip(Bband, np.percentile(Bband, 2.5), np.percentile(Bband, 97.5))

    im = np.array([Rband, Gband, Bband])
    im = im * 255 / np.amax(im)
    im = np.moveaxis(im, 0, -1)

    return im.astype('uint8')

# Cargar los datos desde un geotiff
def load_geotiff(tiff_path):
        
    # Cargar el geotiff
    dataset = rasterio.open(tiff_path)
    
    # Cargar info
    num = dataset.count # numero de bandas
    height, width = dataset.height, dataset.width # shape (Image Resolution)
    crs = dataset.crs # CRS (Coordinate Reference System)
    
    # Corner coordinates
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))    
    xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)
    
    BBox = [[np.amin(lons), np.amax(lats)], 
            [np.amax(lons), np.amin(lats)]]
    
    #Read array values
    if num == 1:
        arr = dataset.read(1)
    else:
        arr = np.zeros((height, width, num))
        for i in range(num):
            arr[:,:,i] = dataset.read(i+1)
            
    return arr, BBox, crs.to_epsg()

def transform_coordinates(pxy, inputEPSG, outputEPSG):
    
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(pxy[0], pxy[1])

    # create input coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)

    # create output coordinate transformation
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)

    # transform point
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    point.Transform(coordTransform)
    
    return point.GetX(), point.GetY()