<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="lith_type" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
      <category value="MCC_metapelite" label="MCC / metapelite" symbol="0" render="true"/>
      <category value="felsic_intrusive" label="Felsic intrusive" symbol="1" render="true"/>
      <category value="sedimentary_cover" label="Sedimentary / cover" symbol="2" render="true"/>
      <category value="mafic_ultramafic" label="Mafic / ultramafic" symbol="3" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="86,180,233,140"/>
          <prop k="outline_color" v="80,80,80,180"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.15"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="1" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="230,159,0,140"/>
          <prop k="outline_color" v="80,80,80,180"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.15"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="2" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="240,228,66,120"/>
          <prop k="outline_color" v="80,80,80,180"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.15"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
      <symbol type="fill" name="3" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="180,180,180,120"/>
          <prop k="outline_color" v="80,80,80,180"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.15"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple"><settings><rendering drawLabels="0"/></settings></labeling>
</qgis>
