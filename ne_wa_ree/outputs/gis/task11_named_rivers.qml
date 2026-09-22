<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="singleSymbol" enableorderby="0" forceraster="0">
    <symbols>
      <symbol type="line" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <prop k="line_color" v="30,90,150,240"/>
          <prop k="line_style" v="solid"/>
          <prop k="line_width" v="1.15"/>
          <prop k="line_width_unit" v="MM"/>
          <prop k="capstyle" v="round"/>
          <prop k="joinstyle" v="round"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="8" namedStyle="Italic" textColor="40,80,120,255" fieldName="name" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,220"/>
      </text-style>
      <placement placement="2" dist="0.8" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="20" obstacle="1"/>
    </settings>
  </labeling>
</qgis>
