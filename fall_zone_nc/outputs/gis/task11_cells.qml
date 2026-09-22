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
      <category value="skip" label="skip" symbol="6" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="127,0,0,235"/>
          <prop k="outline_color" v="60,0,0,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="1" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="215,48,31,200"/>
          <prop k="outline_color" v="120,10,10,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="2" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="244,109,67,170"/>
          <prop k="outline_color" v="180,50,20,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="3" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="253,174,97,120"/>
          <prop k="outline_color" v="200,120,40,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="4" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="254,224,144,70"/>
          <prop k="outline_color" v="200,170,60,200"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="5" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="255,255,230,25"/>
          <prop k="outline_color" v="180,180,160,120"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="6" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="0,0,0,0"/>
          <prop k="outline_color" v="160,160,160,90"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="14" fontWeight="75" namedStyle="Bold" textColor="255,255,255,255" fieldName="hot_label" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="1.4" bufferColor="0,0,0,255" bufferOpacity="0.85"/>
      </text-style>
      <placement placement="4" dist="1.2" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="20" obstacle="1"/>
    </settings>
  </labeling>
</qgis>
