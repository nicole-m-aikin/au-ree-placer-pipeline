<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="rank_bin" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
      <category value="#1-3" label="#1–3  walk first" symbol="0" render="true"/>
      <category value="#4-6" label="#4–6" symbol="1" render="true"/>
      <category value="#7-10" label="#7–10" symbol="2" render="true"/>
      <category value="#11-15" label="#11–15" symbol="3" render="true"/>
      <category value="#16-20" label="#16–20" symbol="4" render="true"/>
      <category value="#21-28" label="#21–28" symbol="5" render="true"/>
      <category value="skip" label="skip" symbol="6" render="false"/>
    </categories>
    <symbols>
      <symbol type="marker" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="127,0,0,235"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="7.5"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="1" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="215,48,31,200"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="6.2"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="2" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="244,109,67,170"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="5.2"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="3" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="253,174,97,120"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="4.2"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="4" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="254,224,144,70"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="3.4"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="5" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="255,255,230,25"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="2.6"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="6" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="0,0,0,0"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="triangle"/>
          <prop k="size" v="2.0"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="11" fontWeight="75" namedStyle="Bold" textColor="255,255,255,255" fieldName="map_label" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="1.4" bufferColor="0,0,0,255" bufferOpacity="0.85"/>
      </text-style>
      <placement placement="4" dist="1.2" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="20" obstacle="1"/>
    </settings>
  </labeling>
</qgis>
