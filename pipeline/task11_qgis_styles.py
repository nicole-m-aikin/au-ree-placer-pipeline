"""QGIS styles that actually load: categorized on rank_bin (same engine as the
orange/blue map that worked), colored as a hot-to-pale walk-rank ramp.
"""

# Only the top ranks stay hot. Mid ranks go yellow. The rest almost disappear.
BINS = [
    ('#1-3',  '127,0,0,235',    '60,0,0,255',     '7.5',  '#1–3  walk first'),
    ('#4-6',  '215,48,31,200',  '120,10,10,255',  '6.2',  '#4–6'),
    ('#7-10', '244,109,67,170', '180,50,20,255',  '5.2',  '#7–10'),
    ('#11-15','253,174,97,120', '200,120,40,255', '4.2',  '#11–15'),
    ('#16-20','254,224,144,70', '200,170,60,200', '3.4',  '#16–20'),
    ('#21-28','255,255,230,25', '180,180,160,120','2.6',  '#21–28'),
    ('skip',  '0,0,0,0',        '160,160,160,90', '2.0',  'skip'),
]


def _poly(name, fill, outline, width):
    return f'''      <symbol type="fill" name="{name}" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="{fill}"/>
          <prop k="outline_color" v="{outline}"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="{width}"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>'''


def _mark(name, color, size, shape):
    return f'''      <symbol type="marker" name="{name}" clip_to_extent="1" alpha="1">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <prop k="color" v="{color}"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="name" v="{shape}"/>
          <prop k="size" v="{size}"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>'''


def _line_qml(color, width, label_field=None):
    labels = '  <labeling type="simple"><settings><rendering drawLabels="0"/></settings></labeling>'
    if label_field:
        labels = '''  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="8" namedStyle="Italic" textColor="40,80,120,255" fieldName="{field}" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,220"/>
      </text-style>
      <placement placement="2" dist="0.8" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="20" obstacle="1"/>
    </settings>
  </labeling>'''.format(field=label_field)
    return '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="singleSymbol" enableorderby="0" forceraster="0">
    <symbols>
      <symbol type="line" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <prop k="line_color" v="{color}"/>
          <prop k="line_style" v="solid"/>
          <prop k="line_width" v="{width}"/>
          <prop k="line_width_unit" v="MM"/>
          <prop k="capstyle" v="round"/>
          <prop k="joinstyle" v="round"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
{labels}
</qgis>
'''.format(color=color, width=width, labels=labels)


def _categorized(kind):
    cats = []
    syms = []
    for i, (value, fill, outline, size, label) in enumerate(BINS):
        render = 'false' if (value == 'skip' and kind != 'cells') else 'true'
        cats.append(
            '      <category value="{0}" label="{1}" symbol="{2}" render="{3}"/>'
            .format(value, label, i, render)
        )
        if kind in ('cells', 'catchments'):
            width = '0.8' if kind == 'catchments' else '0.35'
            # Catchments: keep imagery visible — stronger outline, lighter fill.
            if kind == 'catchments':
                parts = fill.split(',')
                fill = '{0},{1},{2},70'.format(parts[0], parts[1], parts[2])
            syms.append(_poly(str(i), fill, outline, width))
        else:
            if kind == 'pour_points':
                shape = 'triangle'
            elif kind == 'pan_locations':
                shape = 'circle'
            else:
                shape = 'star'
            sz = size if kind == 'pour_points' else str(max(2.6, float(size) - 2.2))
            if kind == 'pan_locations':
                sz = str(max(3.4, float(size) - 1.2))
            syms.append(_mark(str(i), fill, sz, shape))
    labels = ''
    if kind in ('cells', 'pour_points', 'pan_locations'):
        field = 'hot_label' if kind == 'cells' else 'map_label'
        size = '14' if kind == 'cells' else '11'
        labels = '''  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="{size}" fontWeight="75" namedStyle="Bold" textColor="255,255,255,255" fieldName="{field}" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="1.4" bufferColor="0,0,0,255" bufferOpacity="0.85"/>
      </text-style>
      <placement placement="4" dist="1.2" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="20" obstacle="1"/>
    </settings>
  </labeling>'''.format(size=size, field=field)
    else:
        labels = '  <labeling type="simple"><settings><rendering drawLabels="0"/></settings></labeling>'
    return '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="rank_bin" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
{0}
    </categories>
    <symbols>
{1}
    </symbols>
  </renderer-v2>
{2}
</qgis>
'''.format('\n'.join(cats), '\n'.join(syms), labels)


def _hobby_qml():
    """Diamonds by gold_class. Gazetteer unknown is teal; recoveries go hot."""
    cats = [
        ('blank', '160,160,160,220', '4.0', 'blank (the 0-class)'),
        ('color', '240,228,66,230', '4.4', 'color'),
        ('flake', '230,159,0,230', '4.8', 'flake'),
        ('picker', '213,94,0,240', '5.2', 'picker'),
        ('nugget', '204,0,0,250', '5.6', 'nugget'),
        ('unknown', '0,158,115,220', '4.2', 'unknown / gazetteer'),
    ]
    cat_xml = []
    sym_xml = []
    for i, (value, fill, size, label) in enumerate(cats):
        cat_xml.append(
            '      <category value="{0}" label="{1}" symbol="{2}" render="true"/>'
            .format(value, label, i)
        )
        sym_xml.append(_mark(str(i), fill, size, 'diamond'))
    labels = '''  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="8" namedStyle="Italic" textColor="20,60,50,255" fieldName="name" isExpression="0">
        <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,220"/>
      </text-style>
      <placement placement="0" dist="1.0" distUnits="MM"/>
      <rendering drawLabels="1" maxNumLabels="24" obstacle="1"/>
    </settings>
  </labeling>'''
    return '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="gold_class" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
{0}
    </categories>
    <symbols>
{1}
    </symbols>
  </renderer-v2>
{2}
</qgis>
'''.format('\n'.join(cat_xml), '\n'.join(sym_xml), labels)


def _geology_qml():
    cats = [
        ('MCC_metapelite', '86,180,233,140', 'MCC / metapelite'),
        ('felsic_intrusive', '230,159,0,140', 'Felsic intrusive'),
        ('sedimentary_cover', '240,228,66,120', 'Sedimentary / cover'),
        ('mafic_ultramafic', '180,180,180,120', 'Mafic / ultramafic'),
    ]
    cat_xml = []
    sym_xml = []
    for i, (value, fill, label) in enumerate(cats):
        cat_xml.append(
            '      <category value="{0}" label="{1}" symbol="{2}" render="true"/>'
            .format(value, label, i)
        )
        sym_xml.append(_poly(str(i), fill, '80,80,80,180', '0.15'))
    return '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="lith_type" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
{0}
    </categories>
    <symbols>
{1}
    </symbols>
  </renderer-v2>
  <labeling type="simple"><settings><rendering drawLabels="0"/></settings></labeling>
</qgis>
'''.format('\n'.join(cat_xml), '\n'.join(sym_xml))


LAYER_QML = {
    'cells': _categorized('cells'),
    'catchments': _categorized('catchments'),
    'pour_points': _categorized('pour_points'),
    'nure_spots': _categorized('nure_spots'),
    'pan_locations': _categorized('pan_locations'),
    'hobby_reports': _hobby_qml(),
    'streams': _line_qml('70,130,180,210', '0.45'),
    'named_rivers': _line_qml('30,90,150,240', '1.15', label_field='name'),
    'geology': _geology_qml(),
    'geology_structure': _line_qml('90,60,40,220', '0.35'),
    'lidar_index': '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling">
  <renderer-v2 type="singleSymbol" enableorderby="0" forceraster="0">
    <symbols>
      <symbol type="fill" name="0" clip_to_extent="1" alpha="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="0,0,0,0"/>
          <prop k="outline_color" v="40,40,40,160"/>
          <prop k="outline_style" v="dash"/>
          <prop k="outline_width" v="0.35"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple"><settings><rendering drawLabels="0"/></settings></labeling>
</qgis>
''',
}


def embed_qgis_styles(gpkg_path):
    import sqlite3
    con = sqlite3.connect(gpkg_path)
    con.execute(
        '''CREATE TABLE IF NOT EXISTS layer_styles (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            f_table_catalog TEXT,
            f_table_schema TEXT,
            f_table_name TEXT,
            f_geometry_column TEXT,
            styleName TEXT,
            styleQML TEXT,
            styleSLD TEXT,
            useAsDefault INTEGER,
            description TEXT,
            owner TEXT,
            ui TEXT,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )'''
    )
    con.execute('DELETE FROM layer_styles')
    catalog = gpkg_path.split('/')[-1]
    present = {
        row[0] for row in con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
        )
    }
    notes = {
        'streams': 'D8 channels from the DEM',
        'named_rivers': 'Named rivers from the study-area config',
        'geology': 'USGS SGMC state geology (Horton 2017)',
        'geology_structure': 'SGMC faults and contacts',
        'lidar_index': '3DEP 1 m LiDAR clip footprints',
        'nure_spots': 'NURE stream-sediment grabs — chemistry, not a pan pin',
        'pan_locations': 'Where to pan — slope break / power drop / junction',
        'hobby_reports': 'Pamphlet / opt-in pans — catchment hit-rate, not AUC',
    }
    for layer, qml in LAYER_QML.items():
        if present and layer not in present:
            continue
        desc = notes.get(layer, 'Darker red = walk first')
        con.execute(
            '''INSERT INTO layer_styles
               (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,
                styleName, styleQML, styleSLD, useAsDefault, description, owner)
               VALUES (?, '', ?, 'geom', ?, ?, '', 1, ?, '')''',
            (catalog, layer, layer, qml, desc),
        )
    con.commit()
    con.close()


def _bin_for_rank(v):
    import pandas as pd
    if v is None or pd.isna(v):
        return 'skip'
    r = int(v)
    if r <= 3:
        return '#1-3'
    if r <= 6:
        return '#4-6'
    if r <= 10:
        return '#7-10'
    if r <= 15:
        return '#11-15'
    if r <= 20:
        return '#16-20'
    if r <= 28:
        return '#21-28'
    return 'skip'


def add_rank_label(gdf):
    """rank_bin drives the color ramp. map_label is #1–#10 on pours only."""
    if gdf is None or len(gdf) == 0:
        return gdf
    import pandas as pd
    out = gdf.copy()
    if 'walk_rank' not in out.columns:
        out['rank_bin'] = 'skip'
        out['rank_label'] = None
        out['map_label'] = None
        return out

    def _lab(v):
        if v is None or pd.isna(v):
            return None
        try:
            return '#{0}'.format(int(v))
        except (TypeError, ValueError):
            return None

    out['rank_bin'] = out['walk_rank'].map(_bin_for_rank)
    if 'campaign_class' in out.columns:
        out.loc[out['campaign_class'] == 'skip', 'rank_bin'] = 'skip'
    out['rank_label'] = out['walk_rank'].map(_lab)
    out.loc[out['rank_bin'] == 'skip', 'rank_label'] = None
    out['map_label'] = None
    top = out['walk_rank'].notna() & (out['walk_rank'] <= 10)
    if 'role' in out.columns:
        top = top & (out['role'] == 'pour_point')
    out.loc[top, 'map_label'] = out.loc[top, 'rank_label']
    if 'pan_order' in out.columns:
        top_pan = out['walk_rank'].notna() & (out['walk_rank'] <= 10) & (out['pan_order'] == 1)
        out.loc[top_pan, 'map_label'] = out.loc[top_pan, 'rank_label']
    out['hot_label'] = None
    hottest = out['walk_rank'].notna() & (out['walk_rank'] <= 3)
    out.loc[hottest, 'hot_label'] = out.loc[hottest, 'rank_label']
    return out
