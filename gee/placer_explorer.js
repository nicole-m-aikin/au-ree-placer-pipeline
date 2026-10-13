// Where to pan — Earth Engine Code Editor app.
//
// Pins come from pipeline/gee_export.py. Paste into
// code.earthengine.google.com (project placer-picker-511303), Save, Run.
// The published app (Apps) uses the saved script.
//
// A screen, not a gold claim and not a land-status opinion. Confirm claims on
// BLM MLRS, land status on the master title plat, and rules with the agency.

var PROJECT = 'placer-picker-511303';
var ROOT = 'projects/' + PROJECT + '/assets/placer/';
var PANS_ASSET = ROOT + 'pans';
var PADUS_ASSET = ROOT + 'padus41';
var CLAIMS_ACTIVE = ROOT + 'claims_active';
var CLAIMS_CLOSED = ROOT + 'claims_closed';
var MRDS_ASSET = ROOT + 'mrds_gold';
var WALKS_ASSET = ROOT + 'walks';
var STATE_FIPS = '06';
var ALL = 'All California';
// Change this when you upload a new pins asset.
var LAST_UPDATE = 'October 13, 2026';

var INK = '#1C2B24';
var PAPER = '#F3F6F1';
var CARD = '#FFFFFF';
var LINE = '#D5DDD4';
var MUTED = '#5C6B62';
var MOSS = '#1B7A45';
var AMBER = '#8A5A12';
// Lighter = weaker. Darker green = go there first.
var GREENS = ['c5dcc8', '8fbf8a', '3d9a5f', '1b7a45', '0e4d2c'];
var GREEN_HEX = GREENS.map(function (c) { return '#' + c; });

var MODES = {
  'Go pan here': {
    rank: 'rec_rank_in_county',
    score: 'rec_score',
    blurb: 'BLM or Forest Service land, no active claim, on a creek below known gold. State parks are left off.'
  },
  'Possibly stakeable': {
    rank: 'stake_rank_in_county',
    score: 'stake_score',
    blurb: 'BLM or Forest Service land where a claim might be open. Check BLM before you stake anything.'
  }
};

var WHY = {
  slope_break: 'the creek flattens',
  knickpoint_foot: 'below a drop',
  power_drop: 'the flow slows',
  tributary_junction: 'where creeks meet',
  valley_opens: 'the valley opens',
  near_hot_nure: 'near a strong chemistry sample',
  outlet_fallback: 'on the walked creek'
};

var pans = ee.FeatureCollection(PANS_ASSET);
var counties = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('STATEFP', STATE_FIPS));
var dem = ee.ImageCollection('USGS/3DEP/10m_collection').mosaic().select('elevation');
var hillshade = ee.Terrain.hillshade(dem);

var padusFee = ee.FeatureCollection(PADUS_ASSET);
var padusFeeFallback = ee.FeatureCollection('USGS/GAP/PAD-US/v20/fee');
var claimsActive = ee.FeatureCollection(CLAIMS_ACTIVE);
var claimsClosed = ee.FeatureCollection(CLAIMS_CLOSED);
var mrds = ee.FeatureCollection(MRDS_ASSET);
var walks = ee.FeatureCollection(WALKS_ASSET);
var sgmcUnits = ee.FeatureCollection(ROOT + 'sgmc_units');
var sgmcFaults = ee.FeatureCollection(ROOT + 'sgmc_faults');
var sgmcContacts = ee.FeatureCollection(ROOT + 'sgmc_contacts');
var sgmcBedding = ee.FeatureCollection(ROOT + 'sgmc_bedding');
var minedGround = ee.FeatureCollection(ROOT + 'placer_mined');
var sierraGravel = ee.FeatureCollection(ROOT + 'eocene_gravel');
var nure = ee.FeatureCollection(ROOT + 'nure');

var naip = ee.ImageCollection('USDA/NAIP/DOQQ');
var lidar1m = ee.ImageCollection('USGS/3DEP/1m').select('elevation');
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var fires = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')
  .filter(ee.Filter.gt('Ig_Date', ee.Date('2011-01-01').millis()));

var state = {
  mode: 'Go pan here', county: ALL, topN: 10, pinId: null,
  minScore: 0, skipDry: false, nearRoad: false, skipBurn: false
};
var BURN_SINCE = 2016;
var shown = [];
var highlight = null;
// Survives redraw. Filter clicks only swap the pins; a new county rebuilds
// the map and puts these checks and opacities back.
var layerPrefs = {};
var builtForCounty = null;

var map = ui.Map();
map.setOptions('TERRAIN');
map.style().set('cursor', 'hand');
// Drawing tools steal clicks that should select a pin.
map.setControlVisibility({
  all: true,
  drawingToolsControl: false,
  layerList: true,
  zoomControl: true,
  scaleControl: true,
  mapTypeControl: true,
  fullscreenControl: false
});

function scoreIndex(score) {
  var s = Number(score);
  if (!isFinite(s)) return 0;
  var idx = Math.floor((s - 40) / 15);
  if (idx < 0) return 0;
  if (idx > 4) return 4;
  return idx;
}

function scoreStyle(fc, scoreField, pointSize) {
  return fc.map(function (f) {
    var s = ee.Number(f.get(scoreField));
    var idx = s.subtract(40).divide(15).floor().max(0).min(4);
    return f.set('style', {
      color: 'ffffff',
      fillColor: ee.List(GREENS).get(idx),
      pointSize: pointSize,
      pointShape: 'circle',
      width: 1
    });
  }).style({styleProperty: 'style'});
}

function publicLandLayer(geom) {
  var fee = padusFee.filterBounds(geom);
  var styled = function (fc) {
    var blm = fc.filter(ee.Filter.eq('Mang_Name', 'BLM'))
      .style({color: 'C4A35A', fillColor: 'C4A35A55', width: 0.4});
    var usfs = fc.filter(ee.Filter.eq('Mang_Name', 'USFS'))
      .style({color: '2F6B4F', fillColor: '2F6B4F33', width: 0.4});
    var spr = fc.filter(ee.Filter.eq('Mang_Name', 'SPR'))
      .style({color: '3D7EA6', fillColor: '3D7EA644', width: 0.4});
    return ee.ImageCollection([blm, usfs, spr]).mosaic();
  };
  return ee.Image(ee.Algorithms.If(
    fee.size().gt(0), styled(fee), styled(padusFeeFallback.filterBounds(geom))
  ));
}

function layerByName(name) {
  var layers = map.layers();
  var n = layers.length();
  for (var i = 0; i < n; i++) {
    if (layers.get(i).getName() === name) return layers.get(i);
  }
  return null;
}

function rememberLayers() {
  var layers = map.layers();
  var n = layers.length();
  for (var i = 0; i < n; i++) {
    var layer = layers.get(i);
    var name = layer.getName();
    if (!name || name === 'Selected place') continue;
    var opacity = layer.getOpacity();
    layerPrefs[name] = {
      shown: layer.getShown(),
      opacity: typeof opacity === 'number' ? opacity : null
    };
  }
}

function addKeptLayer(eeObject, vis, name, shown, opacity) {
  var pref = layerPrefs[name];
  if (pref) {
    shown = pref.shown;
    if (typeof pref.opacity === 'number') opacity = pref.opacity;
  }
  map.addLayer(eeObject, vis, name, shown, opacity);
}

function addScoutLayers(geom) {
  var naipRecent = naip.filterBounds(geom).filterDate('2018-01-01', '2026-01-01').mosaic();
  addKeptLayer(naipRecent.select(['R', 'G', 'B']), {min: 0, max: 255},
               'Aerial photo', false);

  var hs1m = ee.Terrain.hillshade(lidar1m.filterBounds(geom).mosaic());
  addKeptLayer(hs1m.clip(geom), {min: 120, max: 255},
               'Detailed hillshade (where flown)', false);

  var water = gsw.clip(geom);
  addKeptLayer(water.updateMask(water.gt(0)),
               {min: 0, max: 100, palette: ['c6dbef', '6baed6', '2171b5', '08306b']},
               'How often water is here', false);

  addKeptLayer(fires.filterBounds(geom)
                 .style({color: '8A5A12', fillColor: '00000000', width: 1.5}),
               {}, 'Fires since 2011', false);

  addKeptLayer(walks.filterBounds(geom)
                 .style({color: '3D7EA6', width: 1.5}),
               {}, 'Creeks we walked', false);
  addKeptLayer(mrds.filterBounds(geom)
                 .style({color: 'C4A35A', pointSize: 3, width: 0}),
               {}, 'Known gold sites', false);
  addKeptLayer(claimsActive.filterBounds(geom)
                 .style({color: '1C2B24', fillColor: '1C2B2422', width: 0.5}),
               {}, 'Active claims', false);
  addKeptLayer(claimsClosed.filterBounds(geom)
                 .style({color: '6B4C7A', fillColor: '6B4C7A33', width: 0.5}),
               {}, 'Claims closed in the last year', false);

  var units = sgmcUnits.filterBounds(geom).map(function (f) {
    return f.set('style', {
      color: 'ffffff',
      fillColor: f.get('fill'),
      width: 0.3
    });
  });
  addKeptLayer(units.style({styleProperty: 'style'}), {}, 'Geologic units', false);
  addKeptLayer(sgmcFaults.filterBounds(geom).style({color: '6B2430', width: 1.1}),
               {}, 'Faults', false);
  addKeptLayer(sgmcContacts.filterBounds(geom).style({color: '4A564E', width: 0.6}),
               {}, 'Geologic contacts', false);
  addKeptLayer(sgmcBedding.filterBounds(geom).style({
    color: '1C2B24', pointSize: 3, width: 0
  }), {}, 'Bedding', false);
  addKeptLayer(sierraGravel.filterBounds(geom).style({
    color: 'C4A35A', fillColor: 'C4A35A55', width: 0.6
  }), {}, 'Old Sierra gravel', false);
  addKeptLayer(minedGround.filterBounds(geom).style({
    color: '8A5A12', fillColor: '8A5A1233', width: 0.8
  }), {}, 'Dredge fields and hydraulic pits', false);
  addKeptLayer(nure.filterBounds(geom).style({
    color: '3D7EA6', pointSize: 3, width: 0
  }), {}, 'Chemistry samples', false);
}

function clearHighlight() {
  if (highlight) {
    map.layers().remove(highlight);
    highlight = null;
  }
}

function highlightPin(p) {
  clearHighlight();
  var mode = MODES[state.mode];
  var color = GREENS[scoreIndex(p[mode.score])];
  var fc = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([Number(p.lon), Number(p.lat)]))
  ]);
  highlight = ui.Map.Layer(
    fc.style({
      color: 'ffffff', fillColor: color, pointSize: 18, width: 3, pointShape: 'circle'
    }),
    {},
    'Selected place'
  );
  map.layers().add(highlight);
}

function viewingAll() {
  return state.county === ALL;
}

function applyFieldFilters(fc, scoreField) {
  if (state.minScore > 0) fc = fc.filter(ee.Filter.gte(scoreField, state.minScore));
  if (state.skipDry) fc = fc.filter(ee.Filter.neq('water_regime', 'dry'));
  if (state.nearRoad) fc = fc.filter(ee.Filter.lt('road_m', 1000));
  if (state.skipBurn) {
    fc = fc.map(function (f) {
      var year = ee.Number(ee.Algorithms.If(f.get('fire_year'), f.get('fire_year'), 0));
      return f.set('burn_recent', year.gte(BURN_SINCE));
    }).filter(ee.Filter.eq('burn_recent', 0));
  }
  return fc;
}

function filtersOn() {
  return state.minScore > 0 || state.skipDry || state.nearRoad || state.skipBurn;
}

function redraw(fitCounty) {
  var mode = MODES[state.mode];
  var wholeState = viewingAll();
  var county = wholeState ? counties : counties.filter(ee.Filter.eq('NAME', state.county));
  var inView = wholeState ? pans : pans.filter(ee.Filter.eq('county', state.county));
  var eligible = applyFieldFilters(
    inView.filter(ee.Filter.notNull([mode.rank])), mode.score);
  var others = inView.filter(ee.Filter.notNull([mode.rank]).not());
  if (state.mode === 'Go pan here') {
    var park = ee.Filter.or(
      ee.Filter.eq('access_type', 'state_park'),
      ee.Filter.eq('manager_name', 'SPR')
    );
    others = others.merge(eligible.filter(park));
    eligible = eligible.filter(park.not());
  }
  var geom = wholeState ? county.geometry() : county.geometry().buffer(5000);
  var places = scoreStyle(eligible, mode.score, wholeState ? 6 : 11);
  var otherPlaces = others.style({
    color: 'ffffff', fillColor: '9aa89f', pointSize: 4, width: 1
  });
  var countyChanged = builtForCounty !== state.county || map.layers().length() === 0;

  if (countyChanged) {
    rememberLayers();
    map.layers().reset();
    highlight = null;
    if (!wholeState) {
      addKeptLayer(hillshade.clip(geom), {min: 120, max: 255}, 'Hillshade', false, 0.45);
    }
    addKeptLayer(publicLandLayer(geom), {},
                 'Public land — tan BLM, green Forest Service, blue state park', true, 0.9);
    addScoutLayers(geom);
    if (!wholeState) {
      addKeptLayer(county.style({color: '1C2B24', fillColor: '00000000', width: 1.5}),
                   {}, state.county + ' County');
    }
    addKeptLayer(otherPlaces, {}, 'Other places, not on this list', false);
    addKeptLayer(places, {}, 'Places to check', true);
    builtForCounty = state.county;
  } else {
    var placesLayer = layerByName('Places to check');
    var othersLayer = layerByName('Other places, not on this list');
    if (placesLayer) placesLayer.setEeObject(places);
    else addKeptLayer(places, {}, 'Places to check', true);
    if (othersLayer) othersLayer.setEeObject(otherPlaces);
    else addKeptLayer(otherPlaces, {}, 'Other places, not on this list', false);
    if (!state.pinId) clearHighlight();
  }
  if (fitCounty) map.centerObject(county, wholeState ? 6 : 9);

  modeBlurb.setValue(mode.blurb);
  countLabel.setValue(
    (state.minScore > 0 ? state.minScore + ' and up. ' : '') + 'Showing ' + state.topN);
  fillList(eligible, mode);
}

var panel = ui.Panel({
  style: {width: '420px', padding: '16px 16px 12px 16px', backgroundColor: PAPER}
});

function onPaper(style) {
  var next = {backgroundColor: PAPER};
  if (style) {
    for (var key in style) next[key] = style[key];
  }
  return next;
}

panel.add(ui.Label('Where to pan', onPaper({
  fontSize: '26px', fontWeight: 'bold', color: INK, margin: '0 0 4px 0',
  fontFamily: 'Georgia, serif'
})));
panel.add(ui.Label(
  'California creeks below known gold. Scores are out of 100. Darker green means go there first.',
  onPaper({fontSize: '13px', color: INK, margin: '0 0 10px 0'})
));

function legendRow() {
  var row = ui.Panel({
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {margin: '0 0 14px 0', backgroundColor: PAPER}
  });
  var labels = ['under 55', '55–69', '70–84', '85–99', '100'];
  GREEN_HEX.forEach(function (hex, i) {
    var bit = ui.Panel({
      style: {margin: '0 16px 0 0', backgroundColor: PAPER}
    });
    bit.add(ui.Label('\u00a0', {
      backgroundColor: hex, fontSize: '4px', padding: '3px 14px', margin: '0 0 4px 0'
    }));
    bit.add(ui.Label(labels[i], onPaper({fontSize: '10px', color: MUTED, margin: '0'})));
    row.add(bit);
  });
  return row;
}
panel.add(legendRow());

var modeSelect = ui.Select({
  items: Object.keys(MODES), value: state.mode,
  style: {stretch: 'horizontal', margin: '2px 0 4px 0'},
  onChange: function (v) { state.mode = v; state.pinId = null; redraw(false); }
});
var countySelect = ui.Select({
  placeholder: 'Loading counties…',
  style: {stretch: 'horizontal', margin: '2px 0 4px 0'},
  onChange: function (v) { state.county = v; state.pinId = null; redraw(true); }
});
var topSlider = ui.Slider({
  min: 5, max: 25, step: 5, value: state.topN, style: {stretch: 'horizontal'},
  onChange: function (v) { state.topN = v; redraw(false); }
});
var modeBlurb = ui.Label('', onPaper({fontSize: '13px', color: INK, margin: '0 0 10px 0'}));
var countLabel = ui.Label('Showing 10', onPaper({fontSize: '12px', color: MUTED, margin: '0 0 8px 0'}));
var listPanel = ui.Panel({style: {backgroundColor: PAPER, margin: '0 0 8px 0'}});
var detailPanel = ui.Panel({
  style: {backgroundColor: CARD, padding: '12px', border: '1px solid ' + LINE, margin: '4px 0 10px 0'}
});

function fieldLabel(text) {
  return ui.Label(text, onPaper({fontSize: '12px', color: MUTED, margin: '8px 0 0 0'}));
}
panel.add(fieldLabel('Looking for'));
panel.add(modeSelect);
panel.add(modeBlurb);
panel.add(fieldLabel('Where'));
panel.add(countySelect);
panel.add(fieldLabel('Minimum score'));
var scoreLabel = ui.Label('Any score', onPaper({
  fontSize: '12px', color: MUTED, margin: '0 0 2px 0'
}));
var scoreSlider = ui.Slider({
  min: 0, max: 100, step: 5, value: 0, style: {stretch: 'horizontal'},
  onChange: function (v) {
    state.minScore = v;
    state.pinId = null;
    scoreLabel.setValue(v > 0 ? v + ' and up' : 'Any score');
    redraw(false);
  }
});
panel.add(scoreSlider);
panel.add(scoreLabel);
panel.add(fieldLabel('How many places'));
panel.add(topSlider);
function filterBox(label, key, checked) {
  return ui.Checkbox({
    label: label,
    value: !!checked,
    style: {fontSize: '13px', color: INK, margin: '2px 0', padding: '0', backgroundColor: PAPER},
    onChange: function (v) { state[key] = v; state.pinId = null; redraw(false); }
  });
}
panel.add(fieldLabel('Narrow the list'));
panel.add(filterBox('Only where a satellite saw water', 'skipDry', false));
panel.add(filterBox('Within 1 km of a road', 'nearRoad'));
panel.add(filterBox('Skip burns since 2016', 'skipBurn'));
panel.add(countLabel);
panel.add(listPanel);
panel.add(detailPanel);
panel.add(ui.Label(
  'The number is mostly the shape of the creek and the gold workings upstream. ' +
  'A creek on the old Sierra gravel gets a few extra points. A spot in an old ' +
  'dredge field or hydraulic pit is cut down.',
  onPaper({fontSize: '12px', color: MUTED, margin: '8px 0 6px 0'})
));
panel.add(ui.Label(
  'A satellite often misses a small creek, or one under trees. No water on the ' +
  'pin does not make it a dry wash, and it does not change the score. The checkbox ' +
  'only hides pins where no standing water was seen.',
  onPaper({fontSize: '12px', color: MUTED, margin: '0 0 6px 0'})
));
panel.add(ui.Label(
  'A claim that closed nearby shows on Possibly stakeable. It does not make the creek a better place to pan.',
  onPaper({fontSize: '12px', color: MUTED, margin: '0 0 8px 0'})
));
panel.add(ui.Label(
  'Before you go, confirm the claim and ask the land manager. This list is a screen, not permission.',
  onPaper({fontSize: '12px', color: AMBER, margin: '4px 0'})));
panel.add(ui.Label('Check claims on the BLM map', onPaper({
  fontSize: '13px', color: MOSS, margin: '0 0 8px 0'
}), 'https://mlrs.blm.gov/s/research-map'));
panel.add(ui.Label(
  'Layers, top right: aerial photo, geology, faults, contacts, bedding, Sierra gravel, mined ground, chemistry, fires, walked creeks, known gold, claims. Off until you turn them on.',
  onPaper({fontSize: '11px', color: MUTED, margin: '0 0 6px 0'})));
panel.add(ui.Label(
  'Gold sites, mined ground, and land maps are public U.S. Geological Survey and Bureau of Land Management records.',
  onPaper({fontSize: '11px', color: MUTED, margin: '0 0 6px 0'})));
panel.add(ui.Label('Last update: ' + LAST_UPDATE, onPaper({
  fontSize: '11px', color: MUTED, margin: '0'
})));

function fmt(v, digits) {
  if (v === null || v === undefined || v === '') return '';
  return typeof v === 'number' ? v.toFixed(digits === undefined ? 1 : digits) : String(v);
}

function landName(p) {
  var m = String(p.manager_name || p.access_type || '').toUpperCase();
  if (m === 'USFS' || m === 'usfs') return 'Forest Service';
  if (m === 'BLM' || m === 'blm') return 'BLM';
  if (m === 'SPR' || m === 'state_park') return 'State park';
  if (p.access_type === 'usfs') return 'Forest Service';
  if (p.access_type === 'blm') return 'BLM';
  if (p.access_type === 'state_park') return 'State park';
  if (p.claim_status === 'claimed' || p.access_type === 'claimed') {
    return p.manager_name ? String(p.manager_name) : 'Active claim';
  }
  return p.manager_name || 'Land manager unknown';
}

function whySentence(why) {
  if (!why) return '';
  var parts = String(why).split(';').map(function (bit) {
    return WHY[bit] || bit.replace(/_/g, ' ');
  });
  return parts.join(', ');
}

function buttonStyle(p, mode, selected) {
  var hex = GREEN_HEX[scoreIndex(p[mode.score])];
  return {
    stretch: 'horizontal', margin: '4px 0', padding: '0',
    backgroundColor: PAPER, color: INK, fontSize: '13px', textAlign: 'left',
    border: '2px solid ' + (selected ? INK : hex)
  };
}

function countyFull(name) {
  var n = String(name || '');
  if (!n) return '';
  return n.slice(-7) === ' County' ? n : n + ' County';
}

function countyShort(name) {
  var full = countyFull(name);
  return full ? full.slice(0, -7) + ' Co.' : '';
}

function statePark(p) {
  var access = String(p.access_type || '');
  var manager = String(p.manager_name || '').toUpperCase();
  return access === 'state_park' || manager === 'SPR';
}

function finiteNum(v) {
  if (v === null || v === undefined || v === '') return null;
  var n = Number(v);
  return isFinite(n) ? n : null;
}

function isMined(p) {
  return p.on_mined === true || p.on_mined === 1 || p.on_mined === 'True' || p.on_mined === '1';
}

function gravelPoints(km) {
  if (km === null || km > 5) return 0;
  return 10 * Math.exp(-Math.max(km, 0) / 2);
}

// source_c is not on the pins asset. Recover it from the published score.
function goldOutOfOne(p) {
  var trap = finiteNum(p.trap_score);
  var score = finiteNum(p.evidence_score);
  if (trap === null || score === null) return null;
  var chem = finiteNum(p.chem_c);
  var hasChem = chem !== null;
  var pre = isMined(p) ? score * 2 : score;
  var base = pre - gravelPoints(finiteNum(p.channel_km));
  var t = Math.max(0, Math.min(1, trap / 8.5));
  var den = 0.45 + 0.35 + (hasChem ? 0.20 : 0);
  var chemTerm = hasChem ? 0.20 * chem : 0;
  var s = (base / 100 * den - 0.45 * t - chemTerm) / 0.35;
  if (!isFinite(s)) return null;
  return Math.max(0, Math.min(1, s));
}

function distPhrase(km) {
  if (km < 0.1) return 'on this spot';
  return 'nearest ' + fmt(km, 1) + ' km';
}

function cautionText(note) {
  return String(note).split(';').map(function (bit) {
    bit = bit.replace(/^\s+|\s+$/g, '');
    if (!bit) return '';
    if (bit.toLowerCase().indexOf('critical environmental concern') >= 0) {
      return 'Inside an ACEC: extra restrictions likely.';
    }
    return bit.charAt(0).toUpperCase() + bit.slice(1) + '.';
  }).filter(function (bit) { return bit; }).join(' ');
}

function listLabel(p, mode, place) {
  var bits = viewingAll()
    ? [countyShort(p.county), fmt(p[mode.score], 0), landName(p)]
    : ['#' + place, fmt(p[mode.score], 0), landName(p)];
  if (p.water_regime === 'seasonal' || p.water_regime === 'year_round') {
    bits.push('water seen from space');
  }
  if (p.best_months) bits.push(p.best_months);
  if (p.recent_closed_claim && mode.rank === 'stake_rank_in_county') {
    bits.push('recently closed claim');
  }
  return bits.join('  —  ');
}

function addLine(text, style) {
  if (!text) return;
  detailPanel.add(ui.Label(text, style || {fontSize: '13px', color: INK, margin: '2px 0'}));
}

function addScoreBreakdown(p) {
  var because = whySentence(p.why);
  var trap = finiteNum(p.trap_score);
  if (trap !== null) {
    addLine('Creek shape: ' + fmt(trap, 1) + ' out of 8.5'
            + (because ? '. ' + because.charAt(0).toUpperCase() + because.slice(1) : '')
            + '.');
  }
  var gold = goldOutOfOne(p);
  var nUp = Number(p.n_upstream_gold);
  var up = finiteNum(p.upstream_gold_km);
  var goldLine = 'Gold workings: ' + (gold === null ? 'left out' : fmt(gold, 2) + ' out of 1') + '.';
  if (nUp > 0 && up !== null) {
    goldLine += ' ' + nUp + ' upstream, ' + distPhrase(up) + '.';
  } else if (finiteNum(p.nearest_gold_km) !== null) {
    var nearKm = p.nearest_gold_km;
    goldLine += nearKm < 0.1
      ? ' On this spot'
      : ' Nearest known gold is ' + fmt(nearKm, 1) + ' km off this creek';
    if (p.seed_name) goldLine += ' (' + p.seed_name + ')';
    goldLine += '.';
  } else {
    goldLine += ' None mapped upstream on this creek.';
  }
  addLine(goldLine);
  if (up !== null && up < 0.1 && !isMined(p)) {
    addLine('On this spot is a mapped gold site. The pin is not in a dredge field or hydraulic pit, so the score is not cut.');
  }
  var ch = finiteNum(p.channel_km);
  var gravel = fmt(gravelPoints(ch), 1) + ' out of 10';
  if (ch !== null && ch <= 5) {
    addLine('Old Sierra gravel: ' + gravel + '.');
  } else {
    addLine('Old Sierra gravel: ' + gravel + '. Too far to add points.');
  }
  var chem = finiteNum(p.chem_c);
  if (chem === null) addLine('Chemistry: left out. No sample in range.');
  else addLine('Chemistry: ' + fmt(chem, 2) + ' out of 1.');
  addLine(isMined(p)
    ? 'Weighted sum, scaled to 100, then cut in half.'
    : 'Weighted sum, scaled to 100.');
  if (isMined(p)) {
    addLine('Inside mapped mined ground'
            + (p.mined_name ? ' (' + p.mined_name + ')' : '')
            + ', so the score is cut in half.',
            {fontSize: '13px', color: AMBER, margin: '2px 0', backgroundColor: CARD});
  }
}

function addRockLine(p) {
  var wanted = p.pin_id;
  var rockLabel = ui.Label('', {
    fontSize: '13px', color: INK, margin: '2px 0', backgroundColor: CARD, shown: false
  });
  detailPanel.add(rockLabel);
  var pt = ee.Geometry.Point([Number(p.lon), Number(p.lat)]);
  var elevPin = dem.reduceRegion({
    reducer: ee.Reducer.first(), geometry: pt, scale: 10, maxPixels: 1e6
  }).get('elevation');
  var withDist = sgmcUnits.filterBounds(pt.buffer(8000)).map(function (f) {
    return f.set('dist_m', f.geometry().distance(pt, 50));
  });
  var under = withDist.filter(ee.Filter.lte('dist_m', 40));
  var host = ee.Feature(ee.Algorithms.If(
    under.size().gt(0), under.sort('dist_m').first(), withDist.sort('dist_m').first()));
  var bedrock = withDist
    .filter(ee.Filter.stringStartsWith('lith', 'Unconsolidated').not())
    .sort('dist_m');
  var hostLith = ee.String(ee.Algorithms.If(host.get('lith'), host.get('lith'), ''));
  var chosen = ee.Feature(ee.Algorithms.If(
    hostLith.index('Unconsolidated').gte(0).and(bedrock.size().gt(0)),
    bedrock.first(),
    host
  ));
  var dist = ee.Number(chosen.get('dist_m'));
  var sample = chosen.geometry().intersection(pt.buffer(dist.add(150)), 50);
  var elevRock = dem.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: ee.Geometry(ee.Algorithms.If(dist.lt(100), pt, sample)),
    scale: 30,
    maxPixels: 1e6,
    bestEffort: true
  }).get('elevation');
  ee.Dictionary({
    unit: chosen.get('unit'),
    dist_m: dist,
    elev_pin: elevPin,
    elev_rock: elevRock,
    n: withDist.size()
  }).evaluate(function (d, err) {
    if (state.pinId !== wanted) return;
    if (err || !d || !d.unit) return;
    var km = Number(d.dist_m) / 1000;
    var place = 'at this pin';
    if (km >= 0.1) {
      var rise = Number(d.elev_rock) - Number(d.elev_pin);
      var dir = 'away';
      if (rise > 15) dir = 'upslope';
      else if (rise < -15) dir = 'downslope';
      place = fmt(km, 1) + ' km ' + dir;
    }
    rockLabel.setValue(String(d.unit) + ', ' + place + '.');
    rockLabel.style().set('shown', true);
  });
}

function showDetail(p) {
  state.pinId = p.pin_id;
  var mode = MODES[state.mode];
  detailPanel.clear();
  var rank = finiteNum(p[mode.rank]);
  detailPanel.add(ui.Label(
    (rank === null ? '' : '#' + fmt(rank, 0) + ' in ') + countyFull(p.county), {
    fontSize: '18px', fontWeight: 'bold', color: INK, margin: '0 0 4px 0',
    fontFamily: 'Georgia, serif', backgroundColor: CARD
  }));
  addLine('Score ' + fmt(p[mode.score], 0) + '.',
          {fontSize: '13px', color: MUTED, margin: '0 0 4px 0', backgroundColor: CARD});
  addScoreBreakdown(p);

  var claim = p.claim_status === 'claimed'
    ? 'Active claim on this pin.'
    : 'No active claim on this pin.';
  addLine(landName(p) + '. ' + claim);
  if (statePark(p)) {
    addLine('State park. Most California state parks do not allow panning. A few posted spots do. Check this park before you go.',
            {fontSize: '13px', color: AMBER, margin: '2px 0', backgroundColor: CARD});
  }
  if (p.best_months) addLine('Best months: ' + p.best_months);
  if (p.water_regime === 'dry') {
    addLine('No standing water showed up in the satellite record. A small creek, or one under trees, often looks dry from space.',
            {fontSize: '13px', color: MUTED, margin: '2px 0', backgroundColor: CARD});
  } else if (p.water_regime === 'seasonal' || p.water_regime === 'year_round') {
    addLine('A satellite saw standing water here. That is a wide, open patch of water, not a measure of how the creek flows.');
  }
  if (p.elev_m !== null && p.elev_m !== undefined && p.elev_m !== '') {
    addLine(fmt(p.elev_m, 0) + ' m elevation.');
  }
  if (p.road_m !== null && p.road_m !== undefined && p.road_m !== '') {
    var road = Number(p.road_m);
    addLine(road >= 5000 ? 'No mapped road within 5 km.' : 'About ' + fmt(road, 0) + ' m to a mapped road.');
  }
  if (p.fire_year) addLine('A fire burned within 2 km in ' + p.fire_year + '.');
  addRockLine(p);
  if (p.withdrawal_hint) addLine('Possible withdrawal: ' + p.withdrawal_hint,
                                 {fontSize: '13px', color: AMBER, margin: '2px 0', backgroundColor: CARD});
  if (p.caution_note) addLine(cautionText(p.caution_note),
                              {fontSize: '13px', color: AMBER, margin: '2px 0', backgroundColor: CARD});
  if (p.recent_closed_claim) addLine('A nearby claim closed recently: ' + p.recent_closed_claim);
  if (p.rules_note) addLine(p.rules_note, {fontSize: '12px', color: MUTED, margin: '6px 0 8px 0', backgroundColor: CARD});

  detailPanel.add(ui.Label('Open in Google Maps', {
    fontSize: '14px', color: '#ffffff', backgroundColor: MOSS,
    padding: '10px 12px', margin: '8px 0 0 0', textAlign: 'center'
  }, 'https://www.google.com/maps?q=' + p.lat + ',' + p.lon));

  paintList();
  highlightPin(p);
  map.setCenter(Number(p.lon), Number(p.lat), 15);
}

function paintList() {
  var mode = MODES[state.mode];
  listPanel.clear();
  if (!shown.length) {
    listPanel.add(ui.Label(
      filtersOn()
        ? 'Nothing matches. Lower the minimum score or uncheck a filter.'
        : (viewingAll()
          ? 'No places to check yet.'
          : 'No places to check in ' + state.county + ' for this list.'),
      onPaper({fontSize: '13px', color: MUTED, margin: '4px 0'})));
    return;
  }
  shown.forEach(function (p, i) {
    listPanel.add(ui.Button({
      label: listLabel(p, mode, i + 1),
      style: buttonStyle(p, mode, p.pin_id === state.pinId),
      onClick: function () { showDetail(p); }
    }));
  });
}

function fillList(eligible, mode) {
  listPanel.clear();
  detailPanel.clear();
  detailPanel.add(ui.Label('Click a green dot on the map, or a row in the list.', {
    fontSize: '13px', color: MUTED, margin: '0', backgroundColor: CARD
  }));
  listPanel.add(ui.Label('Loading places…', onPaper({fontSize: '13px', color: MUTED})));
  var ordered = viewingAll() ? eligible.sort(mode.score, false) : eligible.sort(mode.rank);
  ordered.limit(state.topN).evaluate(function (fc, err) {
    if (err) {
      listPanel.clear();
      listPanel.add(ui.Label('The list failed to load. ' + err, onPaper({fontSize: '13px', color: AMBER})));
      return;
    }
    shown = (fc.features || []).map(function (f) { return f.properties; });
    paintList();
  });
}

function clickBufferMeters(lat) {
  // Match the visible pin: at county zoom a 12 px dot is more than 400 m wide.
  var zoom = Number(map.getZoom());
  if (!isFinite(zoom)) zoom = 9;
  var metersPerPixel = 156543.03 * Math.cos(lat * Math.PI / 180) / Math.pow(2, zoom);
  return Math.min(12000, Math.max(80, metersPerPixel * 22));
}

map.onClick(function (c) {
  var pt = ee.Geometry.Point([c.lon, c.lat]);
  var near = pt.buffer(clickBufferMeters(c.lat), 1);
  pans.filterBounds(near).map(function (f) {
    return f.set('d', f.geometry().distance(pt, 1));
  }).sort('d').limit(1).evaluate(function (fc) {
    if (fc && fc.features.length) showDetail(fc.features[0].properties);
  });
});

var legend = ui.Panel({
  style: {
    position: 'bottom-right', padding: '8px 10px', margin: '8px',
    backgroundColor: PAPER, border: '1px solid ' + LINE
  },
  widgets: [
    ui.Label('Darker green = go first', onPaper({
      fontSize: '12px', color: INK, margin: '0 0 4px 0'
    })),
    ui.Label('Tan land is BLM. Green land is Forest Service. Blue is a state park.', onPaper({
      fontSize: '11px', color: MUTED, margin: '0'
    }))
  ]
});
map.add(legend);

pans.aggregate_array('county').distinct().sort().evaluate(function (names) {
  var built = (names || []).filter(function (n) { return n; });
  countySelect.items().reset([ALL].concat(built));
  if (state.county !== ALL && built.indexOf(state.county) < 0) state.county = ALL;
  countySelect.setValue(state.county, false);
  countySelect.setPlaceholder(built.length + ' counties');
  redraw(true);
});

ui.root.clear();
ui.root.add(ui.SplitPanel(panel, map));
