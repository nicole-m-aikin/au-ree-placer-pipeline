// Placer explorer — Earth Engine Code Editor app.
//
// Pins come from pipeline/gee_export.py: one row per pan pin, ranked inside
// each county two ways (go pan here / possibly stakeable). Paste this script
// into code.earthengine.google.com, set PANS_ASSET, Run. Apps → New App
// publishes it as a web page.
//
// A screen, not a gold claim and not a land-status opinion. Confirm claims on
// BLM MLRS, land status on the master title plat, and rules with the agency.

var PANS_ASSET = 'projects/placer-picker-511303/assets/placer/pans';
var STATE_FIPS = '06';
var DEFAULT_COUNTY = 'Kern';

var MODES = {
  'Go pan here': {
    rank: 'rec_rank_in_county',
    score: 'rec_score',
    blurb: 'Open public land (BLM / USFS / state park), claim-free, on a ' +
           'trap reach below known gold. Hands-and-pans rules apply.'
  },
  'Possibly stakeable': {
    rank: 'stake_rank_in_county',
    score: 'stake_score',
    blurb: 'BLM or USFS fee land, no active claim nearby, no PAD-US withdrawal ' +
           'hint. "closed:" = a claim BLM closed in the last year.'
  }
};
var PALETTE = ['ffffb2', 'fecc5c', 'fd8d3c', 'f03b20', 'bd0026'];

var pans = ee.FeatureCollection(PANS_ASSET);
var counties = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('STATEFP', STATE_FIPS));
var dem = ee.ImageCollection('USGS/3DEP/10m_collection').mosaic().select('elevation');
var hillshade = ee.Terrain.hillshade(dem);
var padusFee = ee.FeatureCollection('USGS/GAP/PAD-US/v20/fee');

// Scouting-only layers. None of these feed trap_score / evidence_score —
// that ranking stays local (pipeline/county_rank.py, 30 m Copernicus DEM).
// These are map-toggle context for deciding whether to walk a pin.
var naip = ee.ImageCollection('USDA/NAIP/DOQQ');
var lidar1m = ee.ImageCollection('USGS/3DEP/1m').select('elevation');
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var fires = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')
  .filter(ee.Filter.gt('Ig_Date', ee.Date('2011-01-01').millis()));

var state = {mode: 'Go pan here', county: DEFAULT_COUNTY, topN: 10};

// ── Map ────────────────────────────────────────────────────────────────────
var map = ui.Map();
map.setOptions('TERRAIN');
map.style().set('cursor', 'crosshair');

function scoreStyle(fc, scoreField) {
  return fc.map(function (f) {
    var s = ee.Number(f.get(scoreField));
    var idx = s.subtract(40).divide(15).floor().max(0).min(4);
    return f.set('style', {
      color: '000000',
      fillColor: ee.List(PALETTE).get(idx),
      pointSize: 7,
      pointShape: 'circle',
      width: 1
    });
  }).style({styleProperty: 'style'});
}

function publicLandLayer(geom) {
  var fee = padusFee.filterBounds(geom);
  var blm = fee.filter(ee.Filter.eq('Mang_Name', 'BLM'))
    .style({color: 'E69F00', fillColor: 'E69F0040', width: 0.5});
  var usfs = fee.filter(ee.Filter.eq('Mang_Name', 'USFS'))
    .style({color: '009E73', fillColor: '009E7340', width: 0.5});
  var spr = fee.filter(ee.Filter.eq('Mang_Name', 'SPR'))
    .style({color: '56B4E9', fillColor: '56B4E940', width: 0.5});
  return ee.ImageCollection([blm, usfs, spr]).mosaic();
}

function addScoutLayers(geom) {
  // NAIP true color — most recent flight in the window, ~1 m. Good for
  // checking exposed gravel, road access, and structures on a pin.
  var naipRecent = naip.filterBounds(geom).filterDate('2018-01-01', '2026-01-01').mosaic();
  map.addLayer(naipRecent.select(['R', 'G', 'B']), {min: 0, max: 255},
               'NAIP aerial (true color, 2018+)', false);

  // 1 m 3DEP hillshade. Coverage is patchy — blank where no 1 m lidar has
  // been flown; that absence is itself the signal to fall back on 10 m.
  var hs1m = ee.Terrain.hillshade(lidar1m.filterBounds(geom).mosaic());
  map.addLayer(hs1m.clip(geom), {min: 120, max: 255}, '1 m lidar hillshade (3DEP, where flown)', false);

  // JRC occurrence %: how often a cell has held water, 1984–2021. Masked to
  // occurrence > 0 so it only marks wet ground, not the whole scene. Helps
  // tell a perennial reach from a dry wash before walking it.
  var water = gsw.clip(geom);
  map.addLayer(water.updateMask(water.gt(0)),
               {min: 0, max: 100, palette: ['c6dbef', '6baed6', '2171b5', '08306b']},
               'Surface water occurrence % (JRC, 1984–2021)', false);

  // MTBS perimeters, fires since 2011. Burned slopes can shed fresh gravel
  // for a few seasons, but also carry closures — shown either way.
  map.addLayer(fires.filterBounds(geom)
                 .style({color: 'd94701', fillColor: '00000000', width: 1.5}),
               {}, 'Fire perimeters since 2011 (MTBS)', false);
}

function redraw() {
  var mode = MODES[state.mode];
  var county = counties.filter(ee.Filter.eq('NAME', state.county));
  var inCounty = pans.filter(ee.Filter.eq('county', state.county));
  var eligible = inCounty.filter(ee.Filter.notNull([mode.rank]));
  var others = inCounty.filter(ee.Filter.notNull([mode.rank]).not());

  map.layers().reset();
  map.addLayer(hillshade.clip(county.geometry().buffer(5000)),
               {min: 120, max: 255}, 'Hillshade (3DEP 10 m)', true, 0.5);
  map.addLayer(publicLandLayer(county.geometry()), {},
               'Public land (PAD-US fee: BLM / USFS / state park)', true, 0.8);
  addScoutLayers(county.geometry().buffer(5000));
  map.addLayer(county.style({color: '333333', fillColor: '00000000', width: 2}),
               {}, state.county + ' County');
  map.addLayer(others.style({color: '777777', pointSize: 3, width: 0}),
               {}, 'Other pins (not eligible in this mode)', false);
  map.addLayer(scoreStyle(eligible, mode.score), {}, state.mode);
  map.centerObject(county, 9);

  modeBlurb.setValue(mode.blurb);
  fillList(eligible, mode);
}

// ── Panel ──────────────────────────────────────────────────────────────────
var panel = ui.Panel({style: {width: '380px', padding: '8px'}});
panel.add(ui.Label('Placer explorer — California',
                   {fontSize: '20px', fontWeight: 'bold'}));
panel.add(ui.Label(
  'Pins sit on trap-like bends of creeks below gold MRDS sites, scored by ' +
  'trap geometry (45%), distance to known gold (35%) and NURE Au/As ' +
  'chemistry (20%, where sampled). A screen, not a promise of gold.',
  {fontSize: '12px', color: '555555'}));

var modeSelect = ui.Select({
  items: Object.keys(MODES), value: state.mode,
  onChange: function (v) { state.mode = v; redraw(); }
});
var countySelect = ui.Select({
  placeholder: 'Loading counties…',
  onChange: function (v) { state.county = v; redraw(); }
});
var topSlider = ui.Slider({
  min: 5, max: 25, step: 5, value: state.topN, style: {stretch: 'horizontal'},
  onChange: function (v) { state.topN = v; redraw(); }
});
var modeBlurb = ui.Label('', {fontSize: '12px', color: '555555'});
var listPanel = ui.Panel();
var detailPanel = ui.Panel({style: {border: '1px solid #ccc', padding: '4px'}});

panel.add(ui.Label('Mode', {fontWeight: 'bold'}));
panel.add(modeSelect);
panel.add(modeBlurb);
panel.add(ui.Label('County', {fontWeight: 'bold'}));
panel.add(countySelect);
panel.add(ui.Label('How many', {fontWeight: 'bold'}));
panel.add(topSlider);
panel.add(listPanel);
panel.add(ui.Label('Click a pin on the map for details', {fontWeight: 'bold'}));
panel.add(detailPanel);
panel.add(ui.Label(
  'Before you go: confirm claims on BLM MLRS, land status on the master ' +
  'title plat, and panning / dredging rules with the managing agency.',
  {fontSize: '11px', color: '880000'}));
panel.add(ui.Label('BLM MLRS research map', {fontSize: '11px'},
                   'https://mlrs.blm.gov/s/research-map'));
panel.add(ui.Label(
  'Map layer list (top right of the map) has NAIP imagery, 1 m lidar ' +
  'hillshade where flown, surface-water occurrence, and fire perimeters ' +
  'since 2011 — scouting context only, off by default. None of these feed ' +
  'the pin scores above; scoring stays on the 30 m DEM in the pipeline.',
  {fontSize: '11px', color: '555555'}));

function fmt(v, digits) {
  if (v === null || v === undefined || v === '') return '—';
  return typeof v === 'number' ? v.toFixed(digits === undefined ? 1 : digits) : String(v);
}

function showDetail(p) {
  detailPanel.clear();
  var lines = [
    [p.pin_id + '  (' + fmt(p.county) + ' County)', true],
    ['Lat, lon: ' + fmt(p.lat, 5) + ', ' + fmt(p.lon, 5)],
    ['Elevation: ' + fmt(p.elev_m, 0) + ' m   creek: ' + fmt(p.creek_name)],
    ['Evidence score: ' + fmt(p.evidence_score) + '   trap: ' + fmt(p.trap_score) +
     ' (' + fmt(p.why) + ')'],
    ['Nearest gold MRDS: ' + fmt(p.nearest_gold_km) + ' km (' + fmt(p.seed_name) +
     ')   within 2 km: ' + fmt(p.n_gold_mrds_2km, 0)],
    ['NURE Au/As percentile: ' + fmt(p.chem_c, 2)],
    ['Land: ' + fmt(p.access_type) + ' / ' + fmt(p.manager_name) +
     '   claim: ' + fmt(p.claim_status)],
    ['Go pan rank: ' + fmt(p.rec_rank_in_county, 0) +
     '   Stakeable rank: ' + fmt(p.stake_rank_in_county, 0) +
     ' (' + fmt(p.stake_status) + ')'],
  ];
  if (p.withdrawal_hint) lines.push(['Withdrawal hint: ' + p.withdrawal_hint]);
  if (p.caution_note) lines.push(['Caution: ' + p.caution_note]);
  if (p.recent_closed_claim) lines.push(['Recently closed claim nearby: ' + p.recent_closed_claim]);
  if (p.rules_note) lines.push(['Rules: ' + p.rules_note]);
  lines.forEach(function (l) {
    detailPanel.add(ui.Label(l[0], {fontSize: '12px', fontWeight: l[1] ? 'bold' : 'normal',
                                    margin: '1px 4px'}));
  });
  detailPanel.add(ui.Label('Open in Google Maps', {fontSize: '12px'},
    'https://www.google.com/maps?q=' + p.lat + ',' + p.lon));
}

function fillList(eligible, mode) {
  listPanel.clear();
  listPanel.add(ui.Label('Loading…', {color: '777777'}));
  eligible.sort(mode.rank).limit(state.topN).evaluate(function (fc, err) {
    listPanel.clear();
    if (err) {
      listPanel.add(ui.Label('Error: ' + err, {color: 'red'}));
      return;
    }
    if (!fc.features.length) {
      listPanel.add(ui.Label(
        'No pins for ' + state.county + ' County in this mode. The county may ' +
        'not be built yet (see statewide/outputs/tables/county_coverage.csv).',
        {color: '777777'}));
      return;
    }
    fc.features.forEach(function (f) {
      var p = f.properties;
      var text = '#' + p[mode.rank] + '  score ' + fmt(p[mode.score], 0) + '  ' +
        fmt(p.manager_name || p.access_type) + '  near ' + fmt(p.seed_name);
      if (p.recent_closed_claim && mode.rank === 'stake_rank_in_county') {
        text += '  closed: ' + p.recent_closed_claim;
      }
      var b = ui.Button({
        label: text, style: {stretch: 'horizontal', margin: '1px 0'},
        onClick: function () {
          map.setCenter(p.lon, p.lat, 15);
          showDetail(p);
        }
      });
      listPanel.add(b);
    });
  });
}

map.onClick(function (c) {
  var pt = ee.Geometry.Point([c.lon, c.lat]);
  pans.filterBounds(pt.buffer(400)).map(function (f) {
    return f.set('d', f.geometry().distance(pt));
  }).sort('d').limit(1).evaluate(function (fc) {
    if (fc && fc.features.length) showDetail(fc.features[0].properties);
  });
});

// ── Boot ───────────────────────────────────────────────────────────────────
ee.Dictionary({
  all: counties.aggregate_array('NAME').sort(),
  built: pans.aggregate_array('county').distinct().sort()
}).evaluate(function (d) {
  var built = d.built.filter(function (n) { return n; });
  var rest = d.all.filter(function (n) { return built.indexOf(n) < 0; });
  countySelect.items().reset(built.concat(rest));
  if (built.indexOf(state.county) < 0 && built.length) state.county = built[0];
  countySelect.setValue(state.county, false);
  countySelect.setPlaceholder(built.length + ' of ' + d.all.length + ' counties built');
  redraw();
});

ui.root.clear();
ui.root.add(ui.SplitPanel(panel, map));
