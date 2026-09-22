<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="gold_class" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
      <category value="blank" label="blank (the 0-class)" symbol="0" render="true"/>
      <category value="color" label="color" symbol="1" render="true"/>
      <category value="flake" label="flake" symbol="2" render="true"/>
      <category value="picker" label="picker" symbol="3" render="true"/>
      <category value="nugget" label="nugget" symbol="4" render="true"/>
      <category value="unknown" label="unknown / gazetteer" symbol="5" render="true"/>
    </categories>
    <symbols>
      <symbol type="marker" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="160,160,160,220"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="diamond"/>
          <prop k="size" v="4.0"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="1" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="240,228,66,230"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="diamond"/>
          <prop k="size" v="4.4"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="2" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="230,159,0,230"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="diamond"/>
          <prop k="size" v="4.8"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="3" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="213,94,0,240"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="diamond"/>
          <prop k="size" v="5.2"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="4" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="204,0,0,250"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="diamond"/>
          <prop k="size" v="5.6"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="5" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="0,158,115,220"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="diamond"/>
          <prop k="size" v="4.2"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="8" namedStyle="Italic" textColor="20,60,50,255" fieldName="name" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,220"/>
      </text-style>
      <placement placement="0" dist="1.0" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="24" obstacle="1"/>
    </settings>
  </labeling>
</qgis>
