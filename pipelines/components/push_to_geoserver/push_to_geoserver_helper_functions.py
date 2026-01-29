# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import glob
import shutil
import geopandas as gpd
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from string import Template
from typing import Optional
from geo.Geoserver import GeoserverException
from jinja2 import Template as Jinja2Template
from gfm_data_processing.common import logger
from gfm_data_processing import raster_data_operations as rdo


######################################################################################################
###  Add to geoserver
######################################################################################################
def add_imagemosaic_to_geoserver(geo, workspace, task_folder, layer_name, retrieved_file_paths):
    file_type = "imagemosaic"
    content_type = "application/zip"

    css = geo.get_coveragestores(workspace=workspace)
    css_names = []
    # print(css["coverageStores"])
    if "coverageStore" in css["coverageStores"]:
        css_names = [i["name"] for i in css["coverageStores"]["coverageStore"]]

    logger.debug(f"Save to geoserver create coverage store for file: {retrieved_file_paths}")

    layer_folder = f"{task_folder}/{layer_name}"
    os.mkdir(layer_folder)
    for f in retrieved_file_paths:
        shutil.copy(f, f"{layer_folder}/")

    with open(f"{layer_folder}/timeregex.properties", "w") as trp:
        trp.write("regex=[0-9]{4}-[0-9]{2}-[0-9]{2}\n")

    with open(f"{layer_folder}/indexer.properties", "w") as ip:
        ip.write(
            "TimeAttribute=time\nSchema=*the_geom:Polygon,location:String,time:java.util.Date\nPropertyCollectors=TimestampFileNameExtractorSPI[timeregex](time)\n"
        )

    logger.debug(glob.glob(f"{layer_folder}/*.properties"))

    shutil.make_archive(layer_folder, "zip", layer_folder)
    shutil.rmtree(layer_folder)

    file_path_for_coverage_store_asset = f"{layer_folder}.zip"

    if layer_name not in css_names:
        # Create coverage store and publish layer
        geo.create_coveragestore(
            layer_name=layer_name,
            path=file_path_for_coverage_store_asset,
            workspace=workspace,
            file_type=file_type,
            content_type=content_type,
        )

        # Publish time dimension for image mosaic
        if file_type.lower() == "imagemosaic":
            tss = publish_time_dimension_to_coveragestore(geo=geo, store_name=layer_name, workspace=workspace)
            logger.debug(f"{tss}: published imagemosaic time dimension")

        css = geo.get_coveragestore(workspace=workspace, coveragestore_name=layer_name)
        return css

    else:
        add_granule_to_imagemosaic(
            geo=geo, path=file_path_for_coverage_store_asset, workspace=workspace, coverage_store=layer_name
        )
        # Publish time dimension for image mosaic
        if file_type.lower() == "imagemosaic":
            tss = publish_time_dimension_to_coveragestore(geo=geo, store_name=layer_name, workspace=workspace)
            logger.debug(f"{tss}: published imagemosaic time dimension")


def add_netcdf_to_geoserver(geo, workspace, file_path, layer_name, coverage_name):
    """
    Save netcdf to geoserver, assuming store has one feature

    Args:
        file_path (str): path to input file
        e.g ./wind_sa/wind_farm/era_5_output_df_v6.nc
        layer_name (str): name for geoserver layer
        e.g my_nc
        coverage_name (str): name of the coverage to be published
        e.g EFLUX

    Output:
        css (dict): dictionary of new coveragestores created in geoserver

    """
    store_format = "netcdf"
    store_type = "coveragestores"

    zip_file_path = pack_files_in_zip(file_paths=[file_path], store_name=layer_name)

    create_genericstore(
        geo=geo,
        path=zip_file_path,
        workspace=workspace,
        store_name=layer_name,
        store_type=store_type,
        store_format=store_format,
    )

    netcdf_coverage_names = get_available_coverages(geo=geo, workspace=workspace, coveragestore_name=layer_name)

    root = ET.fromstring(netcdf_coverage_names)
    coverageNamesElements = root.findall(".//coverageName")
    for coverageNameElement in coverageNamesElements:
        # assuming one feature
        if coverageNameElement is not None:
            native_layer_name = coverageNameElement.text
            if native_layer_name.lower() == coverage_name.lower():
                publish_layer_for_genericstore(
                    geo=geo,
                    native_layer_name=native_layer_name,
                    layer_name=layer_name,
                    store_name=layer_name,
                    store_type=store_type,
                    workspace=workspace,
                )
                logger.debug(f"published netcdf layer: {layer_name} and coverageName: {native_layer_name}")

    css = geo.get_coveragestore(workspace=workspace, coveragestore_name=layer_name)
    return css


def add_vector_to_geoserver(geo, workspace, file_path, layer_name, store_format):
    """
    Save gpkg or shp to geoserver, assuming store has one feature

    Args:
        file_path (str): path to input file
        e.g ./wind_sa/wind_farm/era_5_output_df_v6.gpkg
        layer_name (str): name for geoserver layer
        e.g my_gpkg
        store_format (str): store_format
        e.g gpkg | shp

    Output:
        css (dict): dictionary of new featurestore created in geoserver

    """
    store_type = "datastores"

    if store_format == "gpkg":
        zip_file_path = pack_files_in_zip(file_paths=[file_path], store_name=layer_name)
    elif store_format == "shp":
        zip_file_path = file_path

    create_genericstore(
        geo=geo,
        path=zip_file_path,
        workspace=workspace,
        store_name=layer_name,
        store_type=store_type,
        store_format=store_format,
    )

    gpkg_store_features = get_featuretypes(geo=geo, workspace=workspace, store_name=layer_name)

    root = ET.fromstring(gpkg_store_features)
    featureTypeNamesElements = root.findall(".//featureTypeName")
    for featureTypeNameElement in featureTypeNamesElements:
        # assuming one feature
        if featureTypeNameElement is not None:
            native_layer_name = featureTypeNameElement.text
            publish_layer_for_genericstore(
                geo=geo,
                native_layer_name=native_layer_name,
                layer_name=layer_name,
                store_name=layer_name,
                store_type=store_type,
                workspace=workspace,
            )

    feature_store = geo.get_featurestore(workspace=workspace, store_name=layer_name)
    css = {}
    css["coverageStore"] = feature_store
    return css


# Create zip archive for file to upload generic store
def pack_files_in_zip(file_paths, store_name):
    path_to_first_file = "/".join(file_paths[0].split("/")[0:-1])
    zip_file_path = f"{path_to_first_file}/{store_name}.zip"

    with ZipFile(zip_file_path, mode="w") as archive:
        for filename in file_paths:
            filename_arcname = os.path.basename(filename)
            archive.write(filename=filename, arcname=filename_arcname)
    return zip_file_path


######################################################################################################
###  Compute bounds
######################################################################################################
def compute_raster_bounds(retrieved_file_path):
    return list(rdo.get_raster_bbox(retrieved_file_path))


def compute_vector_bounds(retrieved_file_path):
    bounds_list = []
    gpkg_meta = gpd.read_file(retrieved_file_path)
    if not gpkg_meta.empty:
        first_layer_bounds = gpkg_meta.total_bounds
        bounds_list = first_layer_bounds.tolist()

    return bounds_list


######################################################################################################
###  Create styles
######################################################################################################
def create_raster_style(gp, full_layer_name):
    layer_style_xml = ""
    if gp.get("geoserver_style", {}).get("segmentation"):
        layer_style_xml = get_segmentation_regression_style(full_layer_name, gp["geoserver_style"]["segmentation"])
    elif gp.get("geoserver_style", {}).get("regression"):
        layer_style_xml = get_segmentation_regression_style(full_layer_name, gp["geoserver_style"]["regression"])
    elif gp.get("geoserver_style", {}).get("rgb"):
        layer_style_xml = get_rgb_style(full_layer_name, gp["geoserver_style"]["rgb"])
    return layer_style_xml


def create_vector_style(gp, full_layer_name):
    if gp.get("geoserver_style"):
        gp["geoserver_style"]["layername"] = full_layer_name
        return get_vector_style(gp.get("geoserver_style"))


def get_rgb_style(layerstylename: str, style_dict: object):
    template_sld_rgb_raster = Template(
        """<?xml version="1.0" encoding="UTF-8"?>
    <StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.opengis.net/sld
    http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd" version="1.0.0">
        <NamedLayer>
            <Name>$layername</Name>
            <UserStyle>
                <Title>$layername</Title>
                <FeatureTypeStyle>
                    <Rule>
                        <RasterSymbolizer>
                            <ChannelSelection>
                                $channelselections            
                            </ChannelSelection>
                        </RasterSymbolizer>
                    </Rule>
                </FeatureTypeStyle>
            </UserStyle>
        </NamedLayer>
    </StyledLayerDescriptor>
    """
    )

    channel_entry_template = Template(
        """<$label>
        <SourceChannelName>$channel</SourceChannelName>
        <ContrastEnhancement>
        <Normalize>
            <VendorOption name="algorithm">StretchToMinimumMaximum</VendorOption>
            <VendorOption name="minValue">$minValue</VendorOption>
            <VendorOption name="maxValue">$maxValue</VendorOption>
        </Normalize>
        </ContrastEnhancement>
    </$label>"""
    )

    channelselections = ""
    for x in style_dict:
        x.setdefault("minValue", 0)
        x.setdefault("maxValue", 255)
        x.setdefault("channel", 1)
        x.setdefault("label", "RedChannel")
        channelselections += channel_entry_template.substitute(x)

    return template_sld_rgb_raster.substitute(
        layername=layerstylename,
        channelselections=channelselections,
    )


def get_segmentation_regression_style(layerstylename: str, style_dict: object):
    template_sld_segmentation_raster = Template(
        """<?xml version="1.0" encoding="UTF-8"?>
    <StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.opengis.net/sld
    http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd" version="1.0.0">
    <NamedLayer>
        <Name>$layername</Name>
        <UserStyle>
        <Title>$layername</Title>
        <IsDefault>1</IsDefault>
        <FeatureTypeStyle>
            <Rule>
            <RasterSymbolizer>
                <Opacity>1.0</Opacity>
                <ColorMap>
                    $colormaps
                </ColorMap>
            </RasterSymbolizer>
            </Rule>
        </FeatureTypeStyle>
        </UserStyle>
    </NamedLayer>
    </StyledLayerDescriptor>"""
    )

    colormap_entry_template = Template(
        '<ColorMapEntry color="$color"  opacity="$opacity" quantity="$quantity" label="$label" />'
    )

    colormapentries = ""
    for x in style_dict:
        x.setdefault("color", "#808080")
        x.setdefault("quantity", 0)
        x.setdefault("opacity", 0)
        x.setdefault("label", "Class 1")
        colormapentries += colormap_entry_template.substitute(x)

    return template_sld_segmentation_raster.substitute(
        layername=layerstylename,
        colormaps=colormapentries,
    )


def get_vector_style(style_dict: object):
    template_sld_segmentation_raster = Jinja2Template(
        """<?xml version="1.0" encoding="UTF-8"?>
    <StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.opengis.net/sld
    http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd" version="1.0.0">
    <NamedLayer>
        <Name>{{ layername }}</Name>
        <UserStyle>
        <Title>{{ layername }}</Title>
        {%- if point_style is mapping and point_style %}
        <FeatureTypeStyle>
            <Rule>
                <ogc:Filter>
                    <ogc:PropertyIsEqualTo>
                    <ogc:Function name="dimension">
                        <ogc:Function name="geometry"/>
                    </ogc:Function>
                    <ogc:Literal>0</ogc:Literal>
                    </ogc:PropertyIsEqualTo>
                </ogc:Filter>
                <PointSymbolizer>
                    <Graphic>
                        <Mark>
                            <WellKnownName>{{ point_style.well_known_name }}</WellKnownName>
                            {%- if point_style.fill or point_style.fill_opacity %}
                            <Fill>
                                {%- if point_style.fill is string and point_style.fill %}
                                <CssParameter name="fill">{{ point_style.fill }}</CssParameter>
                                {%- elif point_style.fill is mapping and 'property_name' in point_style.fill %}
                                <CssParameter name="fill">
                                    <ogc:Function name="Interpolate">
                                        <ogc:PropertyName>{{ point_style.fill.property_name }}</ogc:PropertyName>
                                        {%- for property_value in point_style.fill.property_values %}
                                        <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                        <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                        {% endfor -%}
                                        <ogc:Literal>color</ogc:Literal>
                                    </ogc:Function>
                                </CssParameter>
                                {% endif -%}
                                {%- if (point_style.fill_opacity is integer or point_style.fill_opacity is float) and point_style.fill_opacity %}
                                <CssParameter name="fill-opacity">{{ point_style.fill_opacity }}</CssParameter>
                                {%- elif point_style.fill_opacity is mapping and 'property_name' in point_style.fill_opacity %}
                                <CssParameter name="fill-opacity">
                                    <ogc:Function name="Interpolate">
                                        <ogc:PropertyName>{{ point_style.fill_opacity.property_name }}</ogc:PropertyName>
                                        {%- for property_value in point_style.fill_opacity.property_values %}
                                        <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                        <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                        {% endfor -%}
                                    </ogc:Function>
                                </CssParameter>
                                {% endif -%}
                            </Fill>
                            {% endif -%}
                            {%- if point_style.stroke or point_style.stroke_width or point_style.stroke_opacity %}
                            <Stroke>
                                {%- if point_style.stroke is string and point_style.stroke %}
                                <CssParameter name="stroke">{{ point_style.stroke }}</CssParameter>
                                {%- elif point_style.stroke is mapping and 'property_name' in point_style.stroke %}
                                <CssParameter name="stroke">
                                    <ogc:Function name="Interpolate">
                                        <ogc:PropertyName>{{ point_style.stroke.property_name }}</ogc:PropertyName>
                                        {%- for property_value in point_style.stroke.property_values %}
                                        <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                        <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                        {% endfor -%}
                                        <ogc:Literal>color</ogc:Literal>
                                    </ogc:Function>
                                </CssParameter>
                                {% endif -%}
                                {%- if (point_style.stroke_width is integer or point_style.stroke_width is float) and point_style.stroke_width %}
                                <CssParameter name="stroke-width">{{ point_style.stroke_width }}</CssParameter>
                                {%- elif point_style.stroke_width is mapping and 'property_name' in point_style.stroke_width %}
                                <CssParameter name="stroke-width">
                                    <ogc:Function name="Interpolate">
                                        <ogc:PropertyName>{{ point_style.stroke_width.property_name }}</ogc:PropertyName>
                                        {%- for property_value in point_style.stroke_width.property_values %}
                                        <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                        <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                        {% endfor -%}
                                    </ogc:Function>
                                </CssParameter>
                                {% endif -%}
                                {%- if (point_style.stroke_opacity is integer or point_style.stroke_opacity is float) and point_style.stroke_opacity %}
                                <CssParameter name="stroke-opacity">{{ point_style.stroke_opacity }}</CssParameter>
                                {%- elif point_style.stroke_opacity is mapping and 'property_name' in point_style.stroke_opacity %}
                                <CssParameter name="stroke-opacity">
                                    <ogc:Function name="Interpolate">
                                        <ogc:PropertyName>{{ point_style.stroke_opacity.property_name }}</ogc:PropertyName>
                                        {%- for property_value in point_style.stroke_opacity.property_values %}
                                        <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                        <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                        {% endfor -%}
                                    </ogc:Function>
                                </CssParameter>
                                {% endif -%}
                            </Stroke>
                            {% endif -%}
                        </Mark>
                        {%- if (point_style.size is integer or point_style.size is float) and point_style.size %}
                        <Size>{{ point_style.size }}</Size>
                        {%- elif point_style.size is mapping and 'property_name' in point_style.size %}
                        <Size>
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ point_style.size.property_name }}</ogc:PropertyName>
                                {%- for property_value in point_style.size.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                            </ogc:Function>
                        </Size>
                        {% endif -%}
                    </Graphic>
                </PointSymbolizer>
            </Rule>
        </FeatureTypeStyle>
        {% endif -%}
        {%- if line_style is mapping and line_style %}
        <FeatureTypeStyle>
            <Rule>
                <ogc:Filter>
                    <ogc:PropertyIsEqualTo>
                    <ogc:Function name="dimension">
                        <ogc:Function name="geometry"/>
                    </ogc:Function>
                    <ogc:Literal>1</ogc:Literal>
                    </ogc:PropertyIsEqualTo>
                </ogc:Filter>
                <LineSymbolizer>
                    {%- if line_style.stroke or line_style.stroke_width or line_style.stroke_opacity %}
                    <Stroke>
                        {%- if line_style.stroke is string and line_style.stroke %}
                        <CssParameter name="stroke">{{ line_style.stroke }}</CssParameter>
                        {%- elif line_style.stroke is mapping and 'property_name' in line_style.stroke %}
                        <CssParameter name="stroke">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ line_style.stroke.property_name }}</ogc:PropertyName>
                                {%- for property_value in line_style.stroke.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                                <ogc:Literal>color</ogc:Literal>
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                        {%- if (line_style.stroke_width is integer or line_style.stroke_width is float) and line_style.stroke_width %}
                        <CssParameter name="stroke-width">{{ line_style.stroke_width }}</CssParameter>
                        {%- elif line_style.stroke_width is mapping and 'property_name' in line_style.stroke_width %}
                        <CssParameter name="stroke-width">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ line_style.stroke_width.property_name }}</ogc:PropertyName>
                                {%- for property_value in line_style.stroke_width.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                        {%- if (line_style.stroke_opacity is integer or line_style.stroke_opacity is float) and line_style.stroke_opacity %}
                        <CssParameter name="stroke-opacity">{{ line_style.stroke_opacity }}</CssParameter>
                        {%- elif line_style.stroke_opacity is mapping and 'property_name' in line_style.stroke_opacity %}
                        <CssParameter name="stroke-opacity">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ line_style.stroke_opacity.property_name }}</ogc:PropertyName>
                                {%- for property_value in line_style.stroke_opacity.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                    </Stroke>
                    {% endif -%}
                </LineSymbolizer>
            </Rule>
        </FeatureTypeStyle>
        {% endif -%}
        {%- if polygon_style is mapping and polygon_style %}
        <FeatureTypeStyle>
            <Rule>
                <ogc:Filter>
                    <ogc:PropertyIsEqualTo>
                    <ogc:Function name="dimension">
                        <ogc:Function name="geometry"/>
                    </ogc:Function>
                    <ogc:Literal>2</ogc:Literal>
                    </ogc:PropertyIsEqualTo>
                </ogc:Filter>
                <PolygonSymbolizer>
                    {%- if polygon_style.fill or polygon_style.fill_opacity %}
                    <Fill>
                        {%- if polygon_style.fill is string and polygon_style.fill %}
                        <CssParameter name="fill">{{ polygon_style.fill }}</CssParameter>
                        {%- elif polygon_style.fill is mapping and 'property_name' in polygon_style.fill %}
                        <CssParameter name="fill">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ polygon_style.fill.property_name }}</ogc:PropertyName>
                                {%- for property_value in polygon_style.fill.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                                <ogc:Literal>color</ogc:Literal>
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                        {%- if (polygon_style.fill_opacity is integer or polygon_style.fill_opacity is float) and polygon_style.fill_opacity %}
                        <CssParameter name="fill-opacity">{{ polygon_style.fill_opacity }}</CssParameter>
                        {%- elif polygon_style.fill_opacity is mapping and 'property_name' in polygon_style.fill_opacity %}
                        <CssParameter name="fill-opacity">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ polygon_style.fill_opacity.property_name }}</ogc:PropertyName>
                                {%- for property_value in polygon_style.fill_opacity.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                    </Fill>
                    {% endif -%}
                    {%- if polygon_style.stroke or polygon_style.stroke_width or polygon_style.stroke_opacity %}
                    <Stroke>
                        {%- if polygon_style.stroke is string and polygon_style.stroke %}
                        <CssParameter name="stroke">{{ polygon_style.stroke }}</CssParameter>
                        {%- elif polygon_style.stroke is mapping and 'property_name' in polygon_style.stroke %}
                        <CssParameter name="stroke">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ polygon_style.stroke.property_name }}</ogc:PropertyName>
                                {%- for property_value in polygon_style.stroke.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                                <ogc:Literal>color</ogc:Literal>
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                        {%- if (polygon_style.stroke_width is integer or polygon_style.stroke_width is float) and polygon_style.stroke_width %}
                        <CssParameter name="stroke-width">{{ polygon_style.stroke_width }}</CssParameter>
                        {%- elif polygon_style.stroke_width is mapping and 'property_name' in polygon_style.stroke_width %}
                        <CssParameter name="stroke-width">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ polygon_style.stroke_width.property_name }}</ogc:PropertyName>
                                {%- for property_value in polygon_style.stroke_width.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                        {%- if (polygon_style.stroke_opacity is integer or polygon_style.stroke_opacity is float) and polygon_style.stroke_opacity %}
                        <CssParameter name="stroke-opacity">{{ polygon_style.stroke_opacity }}</CssParameter>
                        {%- elif polygon_style.stroke_opacity is mapping and 'property_name' in polygon_style.stroke_opacity %}
                        <CssParameter name="stroke-opacity">
                            <ogc:Function name="Interpolate">
                                <ogc:PropertyName>{{ polygon_style.stroke_opacity.property_name }}</ogc:PropertyName>
                                {%- for property_value in polygon_style.stroke_opacity.property_values %}
                                <ogc:Literal>{{ property_value.value }}</ogc:Literal>
                                <ogc:Literal>{{ property_value.output }}</ogc:Literal>
                                {% endfor -%}
                            </ogc:Function>
                        </CssParameter>
                        {% endif -%}
                    </Stroke>
                    {% endif -%}
                </PolygonSymbolizer>
            </Rule>
        </FeatureTypeStyle>
        {% endif -%}
        </UserStyle>
    </NamedLayer>
    </StyledLayerDescriptor>"""
    )

    return template_sld_segmentation_raster.render(style_dict)


def publish_time_dimension_to_coveragestore(
    geo,
    store_name: Optional[str] = None,
    workspace: Optional[str] = None,
    presentation: Optional[str] = "LIST",
    units: Optional[str] = "ISO8601",
    default_value: Optional[str] = "MINIMUM",
    content_type: str = "application/xml; charset=UTF-8",
):
    """
    Create time dimension in coverage store to publish time series in geoserver.

    Parameters
    ----------
    geo : Geoserver instance
    store_name : str, optional
    workspace : str, optional
    presentation : str, optional
    units : str, optional
    default_value : str, optional
    content_type : str

    Notes
    -----
    More about time support in geoserver WMS you can read here:
    https://docs.geoserver.org/master/en/user/services/wms/time.html
    """

    url = "{0}/rest/workspaces/{1}/coveragestores/{2}/coverages/{2}".format(geo.service_url, workspace, store_name)

    headers = {"content-type": content_type}

    time_dimension_data = (
        "<coverage>"
        "<enabled>true</enabled>"
        "<metadata>"
        "<entry key='time'>"
        "<dimensionInfo>"
        "<enabled>true</enabled>"
        "<presentation>{}</presentation>"
        "<units>{}</units>"
        "<defaultValue>"
        "<strategy>{}</strategy>"
        "</defaultValue>"
        "<nearestMatchEnabled>true</nearestMatchEnabled>"
        "</dimensionInfo>"
        "</entry>"
        "</metadata>"
        "</coverage>".format(presentation, units, default_value)
    )

    r = geo._requests(method="put", url=url, data=time_dimension_data, headers=headers)
    if r.status_code in [200, 201]:
        return "success"
    else:
        raise GeoserverException(r.status_code, r.content)


def add_granule_to_imagemosaic(
    geo,
    path,
    workspace: str,
    coverage_store: str,
):
    """Adds granule to image mosaic coverage store; Data will uploaded to the server.

    Parameters
    ----------
    geo : Geoserver instance
    path : str
    workspace : str
    coverage_store : str
        The name of coveragestore.

    Notes
    -----
    the path to the granule archive file which contains the geotiff
    """

    if path is None:
        raise Exception("You must provide the full path to the granule archive")

    if workspace is None:
        raise Exception("You must provide the workspace for the coveragestore")

    if coverage_store is None:
        raise Exception("You must provide the coverage_store")

    content_type = "application/zip"

    url = f"{geo.service_url}/rest/workspaces/{workspace}/coveragestores/{coverage_store}/file.imagemosaic?recalculate=nativebbox,latlonbbox"

    headers = {"content-type": content_type}

    r = None
    with open(path, "rb") as f:
        r = geo._requests(method="post", url=url, data=f, headers=headers)

        if r.status_code in [201, 202]:
            return "success"
        else:
            raise GeoserverException(r.status_code, r.content)


def get_available_coverages(geo, coveragestore_name: str, workspace: Optional[str] = None):
    """
    Parameters
    ----------
    geo : Geoserver instance
    coveragestore_name : str
    workspace : str


    Returns the store name if it exists.
    """
    payload = {"recurse": "true"}
    if workspace is None:
        workspace = "default"
    url = "{}/rest/workspaces/{}/coveragestores/{}/coverages.xml?list=all".format(
        geo.service_url, workspace, coveragestore_name
    )
    r = geo._requests(method="get", url=url, params=payload)
    if r.status_code == 200:
        return r.content
    else:
        raise GeoserverException(r.status_code, r.content)


def get_featuretypes(geo, workspace: str = None, store_name: str = None):
    """
    Parameters
    ----------
    geo : Geoserver instance
    workspace : str
    store_name : str
    """
    url = "{}/rest/workspaces/{}/datastores/{}/featuretypes.xml?list=all".format(geo.service_url, workspace, store_name)
    r = geo._requests(method="get", url=url)
    if r.status_code == 200:
        return r.content
    else:
        raise GeoserverException(r.status_code, r.content)


def create_genericstore(
    geo,
    path,
    workspace: Optional[str] = None,
    store_name: Optional[str] = None,
    store_type: str = "coveragestores",
    store_format: str = "netcdf",
):
    """Creates either a coveragestore or datastore; Data will uploaded to the server.
    with no configuration on the store. Hence no available layers.
    Parameters
    ----------
    geo : Geoserver instance
    path : str
        This is path to the archive zip, which has the data for upload
    workspace : str, optional
    store_name : str, optional
        The name of coveragestore. If not provided, parsed from the file name.
    store_type : str
        This can either be coveragestores / datastores
    store_format : str
        This can be netcdf, gpkg, shp
    Notes
    -----
    the path to the file
    """

    if path is None:
        raise Exception("You must provide the full path to the raster")

    if workspace is None:
        workspace = "default"

    if store_name is None:
        store_name = os.path.basename(path)
        f = store_name.split(".")
        if len(f) > 0:
            store_name = f[0]

    store_type = store_type.lower()
    store_format = store_format.lower()

    url = f"{geo.service_url}/rest/workspaces/{workspace}/{store_type}/{store_name}/file.{store_format}?configure=none"

    headers = {"content-type": "application/zip", "Accept": "application/json"}

    r = None
    with open(path, "rb") as f:
        r = geo._requests(method="put", url=url, data=f, headers=headers)

        if r.status_code in [200, 201]:
            return "success"
        else:
            raise GeoserverException(r.status_code, r.content)


def publish_layer_for_genericstore(
    geo,
    native_layer_name: str,
    layer_name: str,
    store_name: Optional[str] = None,
    store_type: str = "coveragestores",
    workspace: Optional[str] = None,
    presentation: Optional[str] = "LIST",
    units: Optional[str] = "ISO8601",
    default_value: Optional[str] = "MINIMUM",
):
    """
    Create layer for generic store including configure time dimension for netcdfs coveragestores.
    Parameters
    ----------
    geo : Geoserver instance
    native_layer_name : str
        This is the native name from the coveragestore;
        Use get_featuretypes or get_available_coverages to retrieve the native name
    layer_name : str
    store_name : str, optional
    store_type : str
        This can either be coveragestores / datastores
    workspace : str, optional
    presentation : str, optional
    units : str, optional
    default_value : str, optional
    content_type : str
    Notes
    -----
    More about time support in geoserver WMS you can read here:
    https://docs.geoserver.org/master/en/user/services/wms/time.html
    """

    store_type = store_type.lower()
    url = ""
    configuration_data = ""

    if store_type == "coveragestores":
        url = f"{geo.service_url}/rest/workspaces/{workspace}/coveragestores/{store_name}/coverages"
        configuration_data = f"""
<coverage>
<name>{layer_name}</name>
<title>{layer_name}</title>
<nativeName>{native_layer_name}</nativeName>
<nativeCoverageName>{native_layer_name}</nativeCoverageName>
<enabled>true</enabled>
<metadata>
<entry key='time'>
<dimensionInfo>
<enabled>true</enabled>
<presentation>{presentation}</presentation>
<units>{units}</units>
<defaultValue>
<strategy>{default_value}</strategy>
</defaultValue>
<nearestMatchEnabled>true</nearestMatchEnabled>
</dimensionInfo>
</entry>
</metadata>
</coverage>
        """
    elif store_type == "datastores":
        url = f"{geo.service_url}/rest/workspaces/{workspace}/datastores/{store_name}/featuretypes"
        configuration_data = f"""
<featureType>
<name>{layer_name}</name>
<title>{layer_name}</title>
<nativeName>{native_layer_name}</nativeName>
</featureType>
        """
    headers = {"content-type": "application/xml; charset=UTF-8"}

    r = geo._requests(method="post", url=url, data=configuration_data, headers=headers)
    if r.status_code in [200, 201]:
        return "success"
    else:
        raise GeoserverException(r.status_code, r.content)
