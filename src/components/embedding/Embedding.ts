import d3 from '../../utils/d3-import';
import type {
  PromptUMAPData,
  PromptPoint,
  GridData,
  QuadtreeNode,
  LevelTileDataItem,
  UMAPPointStreamData,
  LevelTileMap,
  TopicData,
  TopicDataJSON,
  DrawnLabel,
  LabelData,
  DataURLs,
  Direction,
  LoaderWorkerMessage,
  TreeWorkerMessage,
  EmbeddingInitSetting,
  WebGLMatrices
} from '../../types/embedding-types';
import type { Size, Padding, Rect, Point } from '../../types/common-types';
import type { FooterStoreValue, SearchBarStoreValue } from '../../stores';
import {
  getFooterStoreDefaultValue,
  getSearchBarStoreDefaultValue
} from '../../stores';
import type { Writable } from 'svelte/store';
import {
  downloadJSON,
  splitStreamTransform,
  parseJSONTransform,
  timeit,
  rgbToHex,
  round,
  rectsIntersect,
  yieldToMain
} from '../../utils/utils';
import * as Labeler from './EmbeddingLabel';
import createRegl from 'regl';
import * as PointDrawer from './EmbeddingPointWebGL';
import * as Controller from './EmbeddingControl';
import { config } from '../../config/config';
import LoaderWorker from './workers/loader?worker';
import TreeWorker from './workers/tree?worker';

const DEBUG = config.debug;
const HOVER_RADIUS = 3;
let handledFooterMessageID = 0;

let DATA_BASE = `${import.meta.env.BASE_URL}data`;
if (import.meta.env.PROD) {
  DATA_BASE = 'https://pub-596951ee767949aba9096a18685c74bd.r2.dev';
}

/**
 * Class for the Embedding view
 */

export class Embedding {
  svg: d3.Selection<HTMLElement, unknown, null, undefined>;
  /** The size of the BBox of the SVG element */
  svgFullSize: Size;
  /** The size of the drawing space of the SVG element */
  svgSize: Size;
  svgPadding: Padding;

  topSvg: d3.Selection<HTMLElement, unknown, null, undefined>;
  topicCanvases: d3.Selection<HTMLElement, unknown, null, undefined>[];

  pointCanvas: d3.Selection<HTMLElement, unknown, null, undefined>;
  pointRegl: createRegl.Regl;
  frontPositionBuffer: createRegl.Buffer | null = null;
  // frontColorBuffer: createRegl.Buffer | null = null;
  frontTextureCoordinateBuffer: createRegl.Buffer | null = null;
  bufferPointSize = 0;

  // Tooltips
  tooltipTop: HTMLElement;
  tooltipBottom: HTMLElement;
  hoverPoint: PromptPoint | null = null;

  xScale: d3.ScaleLinear<number, number, never>;
  yScale: d3.ScaleLinear<number, number, never>;
  component: HTMLElement;
  updateEmbedding: () => void;

  // Zooming
  zoom: d3.ZoomBehavior<HTMLElement, unknown> | null = null;
  initZoomTransform = d3.zoomIdentity;
  curZoomTransform: d3.ZoomTransform = d3.zoomIdentity;
  curZoomLevel = 1;

  // Interactions
  lastMouseClientPosition: Point | null = null;
  hideHighlights = false;

  // User settings
  showContour: boolean;
  showGrid: boolean;
  showPoint: boolean;
  showLabel: boolean;

  // Data
  dataURLs: DataURLs;
  promptPoints: PromptPoint[] = [];
  gridData: GridData | null = null;
  tileData: LevelTileMap | null = null;
  contours: d3.ContourMultiPolygon[] | null = null;
  contoursInitialized = false;
  loadedPointCount = 1;

  // Time
  playingTimeSlider = false;
  timeScale: d3.ScaleTime<number, number, never> | null = null;
  timeFormatter: ((x: Date) => string) | null = null;
  curTime: string | null = null;
  timeTextureMap: Map<string, number> | null = null;
  timeCountMap: Map<string, number> | null = null;
  timeInspectMode = true;

  // Scatter plot
  lastRefillID = 0;
  lsatRefillTime = 0;
  webGLMatrices: WebGLMatrices | null = null;
  curPointWidth = 1;

  // Stores
  footerStore: Writable<FooterStoreValue>;
  footerStoreValue: FooterStoreValue;
  searchBarStore: Writable<SearchBarStoreValue>;
  searchBarStoreValue: SearchBarStoreValue;

  // Display labels
  topicLevelTrees: Map<number, d3.Quadtree<TopicData>> = new Map<
    number,
    d3.Quadtree<TopicData>
  >();
  maxLabelNum = 0;
  curLabelNum = 0;
  userMaxLabelNum = 20;
  lastLabelNames: Map<string, Direction> = new Map();
  lastLabelTreeLevel: number | null = null;
  lastGridTreeLevels: number[] = [];

  // Web workers
  loaderWorker: Worker;
  treeWorker: Worker;

  // Methods implemented in other files
  // Labels
  drawLabels = Labeler.drawLabels;
  layoutTopicLabels = Labeler.layoutTopicLabels;
  addTileIndicatorPath = Labeler.addTileIndicatorPath;
  getIdealTopicTreeLevel = Labeler.getIdealTopicTreeLevel;
  labelNumSliderChanged = Labeler.labelNumSliderChanged;
  mouseoverLabel = Labeler.mouseoverLabel;
  drawTopicGrid = Labeler.drawTopicGrid;
  redrawTopicGrid = Labeler.redrawTopicGrid;
  drawTopicGridFrame = Labeler.drawTopicGridFrame;

  // Points
  initWebGLBuffers = PointDrawer.initWebGLBuffers;
  updateWebGLBuffers = PointDrawer.updateWebGLBuffers;
  drawScatterPlot = PointDrawer.drawScatterPlot;
  initWebGLMatrices = PointDrawer.initWebGLMatrices;
  highlightPoint = PointDrawer.highlightPoint;
  updateHighlightPoint = PointDrawer.updateHighlightPoint;

  // Control
  initTopControlBar = Controller.initTopControlBar;
  timeSliderMouseDownHandler = Controller.timeSliderMouseDownHandler;
  moveTimeSliderThumb = Controller.moveTimeSliderThumb;
  startTimeSliderAnimation = Controller.startTimeSliderAnimation;

  /**
   *
   * @param args Named parameters
   * @param args.component The component
   */
  constructor({
    component,
    updateEmbedding,
    defaultSetting,
    dataURLs,
    footerStore,
    searchBarStore
  }: {
    component: HTMLElement;
    updateEmbedding: () => void;
    defaultSetting: EmbeddingInitSetting;
    dataURLs: DataURLs;
    footerStore: Writable<FooterStoreValue>;
    searchBarStore: Writable<SearchBarStoreValue>;
  }) {
    this.component = component;
    this.updateEmbedding = updateEmbedding;
    this.dataURLs = dataURLs;

    this.footerStore = footerStore;
    this.footerStoreValue = getFooterStoreDefaultValue();

    this.searchBarStore = searchBarStore;
    this.searchBarStoreValue = getSearchBarStoreDefaultValue();

    // Init some properties based on the default setting
    this.showContour = defaultSetting.showContour;
    this.showGrid = defaultSetting.showGrid;
    this.showPoint = defaultSetting.showPoint;
    this.showLabel = defaultSetting.showLabel;

    // Initialize the web worker to load data and deal with the quadtree
    this.loaderWorker = new LoaderWorker();
    this.loaderWorker.onmessage = (e: MessageEvent<LoaderWorkerMessage>) => {
      this.loaderWorkerMessageHandler(e);
    };

    this.treeWorker = new TreeWorker();
    this.treeWorker.onmessage = (e: MessageEvent<TreeWorkerMessage>) => {
      this.treeWorkerMessageHandler(e);
    };

    // Initialize the SVG
    this.svg = d3.select(this.component).select('.embedding-svg');

    this.svgFullSize = { width: 0, height: 0 };
    const svgBBox = this.svg.node()?.getBoundingClientRect();
    if (svgBBox !== undefined) {
      this.svgFullSize.width = svgBBox.width;
      this.svgFullSize.height = svgBBox.height;
    }

    // Fix the svg width and height
    this.svg
      .attr('width', `${this.svgFullSize.width}px`)
      .attr('height', `${this.svgFullSize.height}px`);

    this.svgPadding = {
      top: 0,
      bottom: 0,
      left: 0,
      right: 0
    };

    // We keep the initial drawing region as a square
    const squareCanvasWidth = Math.min(
      this.svgFullSize.width - this.svgPadding.left - this.svgPadding.right,
      this.svgFullSize.height - this.svgPadding.top - this.svgPadding.bottom
    );

    this.svgSize = {
      width: squareCanvasWidth,
      height: squareCanvasWidth
    };

    this.xScale = d3.scaleLinear();
    this.yScale = d3.scaleLinear();

    // Initialize the SVG groups
    this.initSVGGroups();

    // Initialize the top svg element
    this.topSvg = this.initTopSvg();

    // Initialize the canvases
    this.pointCanvas = d3
      .select(this.component)
      .select<HTMLElement>('.embedding-canvas')
      .attr('width', `${this.svgFullSize.width}px`)
      .attr('height', `${this.svgFullSize.height}px`);
    this.pointRegl = createRegl(this.pointCanvas!.node() as HTMLCanvasElement);

    this.topicCanvases = [];
    for (const pos of ['top', 'bottom']) {
      this.topicCanvases.push(
        d3
          .select(this.component)
          .select<HTMLElement>(`.topic-grid-canvas.${pos}`)
          .attr('width', `${this.svgFullSize.width}px`)
          .attr('height', `${this.svgFullSize.height}px`)
          .classed('hidden', !this.showGrid)
      );
    }

    // Register zoom
    this.zoom = d3
      .zoom<HTMLElement, unknown>()
      .extent([
        [0, 0],
        [this.svgSize.width, this.svgSize.height]
      ])
      .scaleExtent([config.layout.zoomScale[0], config.layout.zoomScale[1]])
      .interpolate(d3.interpolate)
      .on('zoom', (g: d3.D3ZoomEvent<HTMLElement, unknown>) => {
        (async () => {
          await this.zoomed(g);
        })();
      })
      .on('end', () => this.zoomEnded());

    this.topSvg.call(this.zoom).on('dblclick.zoom', null);

    this.tooltipTop = document.querySelector('#popper-tooltip-top')!;
    this.tooltipBottom = document.querySelector('#popper-tooltip-bottom')!;

    // Initialize the data
    timeit('Init data', DEBUG);
    this.initData().then(() => {
      timeit('Init data', DEBUG);
      // Initialize the event handler for the top control bars
      this.initTopControlBar();
    });

    this.initStore();
  }

  /**
   * Initialize the top SVG element
   * @returns Top SVG selection
   */
  initTopSvg = () => {
    const topSvg = d3
      .select(this.component)
      .select<HTMLElement>('.top-svg')
      .attr('width', `${this.svgFullSize.width}px`)
      .attr('height', `${this.svgFullSize.height}px`)
      .on('pointermove', e => this.mousemoveHandler(e as MouseEvent))
      .on('mouseleave', () => {
        this.highlightPoint(undefined);
        this.mouseoverLabel(null, null);
      })
      .attr(
        'transform',
        `translate(${this.svgPadding.left}, ${this.svgPadding.top})`
      );

    const topGroup = topSvg.append('g').attr('class', 'top-group');

    topGroup
      .append('rect')
      .attr('class', 'mouse-track-rect')
      .attr('width', this.svgFullSize.width)
      .attr('height', this.svgFullSize.height);

    const topContent = topGroup.append('g').attr('class', 'top-content');

    topContent.append('g').attr('class', 'topics-bottom');
    topContent
      .append('g')
      .attr('class', 'topics')
      .classed('hidden', !this.showLabel);
    topContent.append('g').attr('class', 'topics-top');
    topContent.append('g').attr('class', 'highlights');
    return topSvg;
  };

  initStore = () => {
    this.footerStore.subscribe(value => {
      this.footerStoreValue = value;

      // Handle message requests from the footer zoom buttons
      if (this.footerStoreValue.messageID !== handledFooterMessageID) {
        handledFooterMessageID = this.footerStoreValue.messageID;

        const zoomBox = this.getCurViewingZoomBox();
        const centerX = zoomBox.x + zoomBox.width / 2;
        const centerY = zoomBox.y + zoomBox.height / 2;

        switch (this.footerStoreValue.messageCommand) {
          case 'zoomIn': {
            // Create a zoomIn transform matrix
            const transform = d3.zoomIdentity
              .translate(
                (this.svgFullSize.width + config.layout.searchPanelWidth) / 2,
                (this.svgFullSize.height + config.layout.topBarHeight) / 2
              )
              .scale(
                Math.min(
                  config.layout.zoomScale[1],
                  this.curZoomTransform.k * 2
                )
              )
              .translate(-centerX, -centerY);

            this.hideHighlights = true;
            this.topSvg
              .transition()
              .duration(300)
              .call(selection => this.zoom?.transform(selection, transform))
              .on('end', () => {
                this.hideHighlights = false;
              });

            break;
          }

          case 'zoomOut': {
            // Create a zoomIn transform matrix
            const transform = d3.zoomIdentity
              .translate(
                (this.svgFullSize.width + config.layout.searchPanelWidth) / 2,
                (this.svgFullSize.height + config.layout.topBarHeight) / 2
              )
              .scale(
                Math.max(
                  config.layout.zoomScale[0],
                  this.curZoomTransform.k * 0.5
                )
              )
              .translate(-centerX, -centerY);

            this.hideHighlights = true;
            this.topSvg
              .transition()
              .duration(300)
              .call(selection => this.zoom?.transform(selection, transform))
              .on('end', () => {
                this.hideHighlights = false;
              });

            break;
          }

          case 'zoomReset': {
            this.hideHighlights = true;
            this.topSvg
              .transition()
              .duration(700)
              .call(selection => {
                this.zoom?.transform(selection, this.initZoomTransform);
              })
              .on('end', () => {
                this.hideHighlights = false;
              });
            break;
          }

          case '': {
            break;
          }

          default: {
            console.error(
              'Unknown message',
              this.footerStoreValue.messageCommand
            );
            break;
          }
        }
      }
    });
  };

  /**
   * Load the UMAP data from json.
   */
  initData = async () => {
    // Read the grid data for contour background
    // Await the data to load to get the range for x and y
    const gridData = await d3.json<GridData>(this.dataURLs.grid);

    if (gridData === undefined) {
      throw Error('Fail to load grid data.');
    }
    this.gridData = gridData;

    // Initialize the data scales
    const xRange = this.gridData.xRange;
    const yRange = this.gridData.yRange;

    // Force the plot to be a square
    let xLength = xRange[1] - xRange[0];
    let yLength = yRange[1] - yRange[0];

    if (!this.gridData.padded) {
      // Add padding for the data
      if (xLength < yLength) {
        yRange[0] -= yLength / 50;
        yRange[1] += yLength / 50;
        yLength = yRange[1] - yRange[0];

        xRange[0] -= (yLength - xLength) / 2;
        xRange[1] += (yLength - xLength) / 2;
      } else {
        // Add padding for the data
        xRange[0] -= xLength / 50;
        xRange[1] += xLength / 50;
        xLength = xRange[1] - xRange[0];

        yRange[0] -= (xLength - yLength) / 2;
        yRange[1] += (xLength - yLength) / 2;
      }
    }

    this.xScale = d3
      .scaleLinear()
      .domain(xRange)
      .range([0, this.svgSize.width]);

    this.yScale = d3
      .scaleLinear()
      .domain(yRange)
      .range([this.svgSize.height, 0]);

    this.contours = this.drawContour();

    // Create time scale if the data has time info
    if (this.gridData.timeGrids) {
      const dates: Date[] = [];
      this.timeTextureMap = new Map<string, number>();
      this.timeCountMap = new Map<string, number>();

      let curI = 0;

      for (const key of Object.keys(this.gridData.timeGrids)) {
        let curDate = new Date(key);
        // If the user doesn't specify a time zone, treat the date in UTC
        if (!key.includes('T')) {
          curDate = new Date(key + 'T00:00:00.000Z');
        }
        dates.push(curDate);

        // Create a map to map time string to texture coordinate
        this.timeTextureMap.set(key, curI);
        curI += 1;

        // Initialize the time counter
        this.timeCountMap.set(key, this.gridData.timeCounter![key]);
      }

      // Add an extra key for rows with invalid time
      this.timeTextureMap.set('bad', curI);

      const minDate = d3.min(dates)!;
      const maxDate = d3.max(dates)!;

      this.timeFormatter = d3.utcFormat(this.gridData.timeFormat!);
      this.timeScale = d3
        .scaleUtc()
        .domain([minDate, maxDate])
        .range([0, config.layout.timeSliderWidth]);
      this.curTime = this.timeFormatter(minDate);
    }

    // Tell the tree worker to prepare to add points to the tree
    const treeMessage: TreeWorkerMessage = {
      command: 'initQuadtree',
      payload: {
        xRange,
        yRange,
        groups: [],
        times: this.timeCountMap ? [...this.timeCountMap.keys()] : []
      }
    };
    this.treeWorker.postMessage(treeMessage);

    // Read the topic label data
    const topicPromise = d3
      .json<TopicDataJSON>(this.dataURLs.topic)
      .then(topicData => {
        if (topicData) {
          // Create a quad tree at each level
          for (const level of Object.keys(topicData!.data)) {
            const tree = d3
              .quadtree<TopicData>()
              .x(d => d[0])
              .y(d => d[1])
              .addAll(topicData!.data[level]);
            this.topicLevelTrees.set(parseInt(level), tree);
          }
        } else {
          console.error('Fail to read topic data.');
        }
      });

    // Show topic labels once we have contours and topic data
    Promise.all([topicPromise]).then(() => {
      this.drawTopicGrid();
      this.layoutTopicLabels(this.userMaxLabelNum);

      // Initialize the slider value
      setTimeout(() => {
        (
          this.component.querySelector(
            'input#slider-label-num'
          ) as HTMLInputElement
        ).value = `${this.curLabelNum}`;
      }, 500);
    });

    // Initialize WebGL matrices once we have the scales
    this.initWebGLMatrices();

    // Send the xScale to the footer
    this.footerStoreValue.xScale = this.xScale;
    this.footerStore.set(this.footerStoreValue);
  };

  /**
   * Initialize the groups to draw elements in the SVG.
   */
  initSVGGroups = () => {
    const umapGroup = this.svg
      .append('g')
      .attr('class', 'umap-group')
      .attr(
        'transform',
        `translate(${this.svgPadding.left}, ${this.svgPadding.top})`
      );

    umapGroup
      .append('g')
      .attr('class', 'contour-group')
      .classed('hidden', !this.showContour);
    umapGroup.append('g').attr('class', 'quad-group');
    umapGroup.append('g').attr('class', 'tile-group');
    umapGroup.append('g').attr('class', 'scatter-group');
  };

  /**
   * Initialize a quadtree
   */
  drawQuadtree = () => {
    const rectGroup = this.svg.select('g.quad-group');
    const tree = d3
      .quadtree<PromptPoint>()
      .x(d => d.x)
      .y(d => d.y)
      .extent([
        [this.xScale.domain()[0], this.yScale.domain()[0]],
        [this.xScale.domain()[1], this.yScale.domain()[1]]
      ])
      .addAll(this.promptPoints);

    // Collapse the quadtree into an array of rectangles.
    const nodes: QuadtreeNode[] = [];
    tree.visit((cur_node, x0, y0, x1, y1) => {
      if (cur_node.length === undefined) {
        const curNode = {
          x0,
          x1,
          y0,
          y1
        };
        nodes.push(curNode);
      }
    });

    // Draw the rectangles
    rectGroup
      .selectAll('.quadtree-node')
      .data(nodes)
      .join('rect')
      .attr('class', 'quadtree-node')
      .attr('x', d => this.xScale(d.x0))
      .attr('y', d => this.yScale(d.y1))
      .attr('width', d => this.yScale(d.y0) - this.yScale(d.y1))
      .attr('height', d => this.xScale(d.x1) - this.xScale(d.x0))
      .style('fill', 'none')
      .style('stroke', 'gray')
      .style('stroke-width', 0.4)
      .style('opacity', 0.9);

    // downloadJSON(tree);
  };

  /**
   * Draw the KDE contour in the background.
   */
  drawContour = () => {
    if (this.gridData == null) {
      console.error('Grid data not initialized');
      return null;
    }

    const contourGroup = this.svg.select<SVGGElement>('.contour-group');

    const gridData1D: number[] = [];
    for (const row of this.gridData.grid) {
      for (const item of row) {
        gridData1D.push(item);
      }
    }

    // Linear interpolate the levels to determine the thresholds
    const levels = 12;
    const thresholds: number[] = [];
    const minValue = Math.min(...gridData1D);
    const maxValue = Math.max(...gridData1D);
    const step = (maxValue - minValue) / levels;
    for (let i = 0; i < levels; i++) {
      thresholds.push(minValue + step * i);
    }

    let contours = d3
      .contours()
      .thresholds(thresholds)
      .size([this.gridData.grid.length, this.gridData.grid[0].length])(
      gridData1D
    );

    // Convert the scale of the generated paths
    const contourXScale = d3
      .scaleLinear()
      .domain([0, this.gridData.grid.length])
      .range(this.gridData.xRange);

    const contourYScale = d3
      .scaleLinear()
      .domain([0, this.gridData.grid[0].length])
      .range(this.gridData.yRange);

    contours = contours.map(item => {
      item.coordinates = item.coordinates.map(coordinates => {
        return coordinates.map(positions => {
          return positions.map(point => {
            return [
              this.xScale(contourXScale(point[0])),
              this.yScale(contourYScale(point[1]))
            ];
          });
        });
      });
      return item;
    });

    // Create a new blue interpolator based on d3.interpolateBlues
    // (starting from white here)
    const blueScale = d3.interpolateLab(
      '#ffffff',
      config.colors['light-blue-800']
    );
    const colorScale = d3.scaleSequential(
      d3.extent(thresholds) as number[],
      d => blueScale(d / 1)
    );

    // const purpleScale = d3.interpolateLab(
    //   '#ffffff',
    //   config.colors['pink-900']
    // );
    // colorScale = d3.scaleSequential(d3.extent(thresholds) as number[], d =>
    //   purpleScale(d / 1)
    // );

    // Draw the contours
    contourGroup
      .selectAll('path')
      .data(contours)
      .join('path')
      .attr('fill', d => colorScale(d.value))
      .attr('d', d3.geoPath());

    // Zoom in to focus on the second level of the contour
    // The first level is at 0
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;

    if (contours.length > 1) {
      for (const coord of contours[1].coordinates) {
        for (const coordPoints of coord) {
          for (const point of coordPoints) {
            if (point[0] < x0) x0 = point[0];
            if (point[1] < y0) y0 = point[1];
            if (point[0] > x1) x1 = point[0];
            if (point[1] > y1) y1 = point[1];
          }
        }
      }
    }

    const screenPadding = 20;
    const viewAreaWidth =
      this.svgFullSize.width - config.layout.searchPanelWidth;
    const viewAreaHeight =
      this.svgFullSize.height -
      config.layout.topBarHeight -
      config.layout.footerHeight;

    const initZoomK = Math.min(
      viewAreaWidth / (x1 - x0 + screenPadding),
      viewAreaHeight / (y1 - y0 + screenPadding)
    );

    this.initZoomTransform = d3.zoomIdentity
      .translate(
        (this.svgFullSize.width + config.layout.searchPanelWidth) / 2,
        (this.svgFullSize.height + config.layout.topBarHeight) / 2
      )
      .scale(initZoomK)
      .translate(-(x0 + (x1 - x0) / 2), -(y0 + (y1 - y0) / 2));

    // Trigger the first zoom
    this.topSvg
      .call(selection =>
        this.zoom?.transform(selection, this.initZoomTransform)
      )
      .on('end', () => {
        this.contoursInitialized = true;
      });

    // Double click to reset zoom to the initial viewpoint
    this.topSvg.on('dblclick', () => {
      this.topSvg
        .transition()
        .duration(700)
        .call(selection => {
          this.zoom?.transform(selection, this.initZoomTransform);
        });
    });

    return contours;
  };

  /**
   * Handler for each zoom event
   * @param e Zoom event
   */
  zoomed = async (e: d3.D3ZoomEvent<HTMLElement, unknown>) => {
    const transform = e.transform;
    const scaleChanged = this.curZoomTransform.k !== transform.k;
    this.curZoomTransform = transform;

    // === Task (1) ===
    // Transform the SVG elements
    this.svg.select('.umap-group').attr('transform', `${transform.toString()}`);

    // Transform the top SVG elements
    this.topSvg
      .select('.top-group')
      .attr('transform', `${transform.toString()}`);

    // Transform the visible canvas elements
    if (this.showPoint) {
      if (this.frontPositionBuffer && this.frontTextureCoordinateBuffer) {
        this.drawScatterPlot();
      }
    }

    // Adjust the label size based on the zoom level
    if (this.showLabel) {
      this.layoutTopicLabels(this.userMaxLabelNum);
    }

    // Adjust the canvas grid based on the zoom level
    if (this.showGrid) {
      this.redrawTopicGrid();
    }

    // Adjust the highlighted tile
    if (this.showGrid && this.lastMouseClientPosition && !this.hideHighlights) {
      this.mouseoverLabel(
        this.lastMouseClientPosition.x,
        this.lastMouseClientPosition.y
      );
    }

    // Adjust the highlighted point
    if (
      this.showPoint &&
      this.lastMouseClientPosition &&
      !this.hideHighlights
    ) {
      const { x, y } = this.lastMouseClientPosition;
      this.mouseoverPoint(x, y);
      this.updateHighlightPoint();
    }

    await yieldToMain();

    // === Task (2) ===
    // Update the footer with the new zoom level
    if (scaleChanged) {
      this.footerStoreValue.curZoomTransform = this.curZoomTransform;
      this.footerStore.set(this.footerStoreValue);
    }
  };

  /**
   * Event handler for zoom ended
   */
  zoomEnded = () => {
    // Update the points (the last call during zoomed() might be skipped)
  };

  /**
   * Handle messages from the embedding worker
   * @param e Message event
   */
  loaderWorkerMessageHandler = (e: MessageEvent<LoaderWorkerMessage>) => {
    switch (e.data.command) {
      case 'transferLoadData': {
        // Add these points to the quadtree ASAP
        const treeMessage: TreeWorkerMessage = {
          command: 'updateQuadtree',
          payload: {
            points: e.data.payload.points
          }
        };
        this.treeWorker.postMessage(treeMessage);

        if (e.data.payload.isFirstBatch) {
          // Add the first batch points
          this.promptPoints = e.data.payload.points;
          this.initWebGLBuffers();
          if (this.showPoint) {
            this.drawScatterPlot();
          }

          // TODO: Remove me (testing search panel)
          const results = this.promptPoints.map(d => d.prompt);
          this.searchBarStoreValue.shown = true;
          this.searchBarStoreValue.results = results;
          this.searchBarStore.set(this.searchBarStoreValue);
        } else {
          // Batches after the first batch
          // Add the points to the the prompt point list
          const newPoints = e.data.payload.points;
          for (const point of newPoints) {
            this.promptPoints.push(point);
          }

          // Add the new points to the WebGL buffers
          this.updateWebGLBuffers(newPoints);
          if (this.showPoint) {
            this.drawScatterPlot();
          }

          if (e.data.payload.isLastBatch) {
            console.log('Finished loading all data.');
          }
        }

        // Update the data point count
        this.loadedPointCount = e.data.payload.loadedPointCount;

        // Update the footer
        this.footerStoreValue.numPoints = this.promptPoints.length;
        this.footerStore.set(this.footerStoreValue);
        break;
      }

      default: {
        console.error('Unknown message', e.data.command);
        break;
      }
    }
  };

  /**
   * Handle messages from the embedding worker
   * @param e Message event
   */
  treeWorkerMessageHandler = (e: MessageEvent<TreeWorkerMessage>) => {
    switch (e.data.command) {
      case 'finishInitQuadtree': {
        // Tell the loader worker to start loading data
        // (need to wait to set up the quadtree)
        const message: LoaderWorkerMessage = {
          command: 'startLoadData',
          payload: { url: this.dataURLs.point }
        };
        this.loaderWorker.postMessage(message);
        break;
      }

      case 'finishQuadtreeSearch': {
        if (this.lastMouseClientPosition === null) {
          throw Error('lastMouseClientPosition is null');
        }
        // Check if the closest point is relatively close to the mouse
        const closestPoint = e.data.payload.point;
        const screenPointX = this.curZoomTransform.applyX(
          this.xScale(closestPoint.x)
        );
        const screenPointY = this.curZoomTransform.applyY(
          this.yScale(closestPoint.y)
        );

        const distance = Math.max(
          Math.abs(screenPointX - this.lastMouseClientPosition.x),
          Math.abs(screenPointY - this.lastMouseClientPosition.y)
        );

        const highlightRadius = Math.max(
          10 / this.curZoomTransform.k,
          (config.layout.scatterDotRadius *
            Math.exp(Math.log(this.curZoomTransform.k) * 0.55)) /
            this.curZoomTransform.k
        );

        // Highlight the point if it is close enough to the mouse
        const curHoverRadius = Math.max(
          HOVER_RADIUS,
          highlightRadius * this.curZoomTransform.k
        );

        if (distance <= curHoverRadius) {
          this.highlightPoint(closestPoint);
        } else {
          this.highlightPoint(undefined);
        }
        break;
      }

      default: {
        console.error('Unknown message', e.data.command);
        break;
      }
    }
  };

  /**
   * Start a query for mouse overed point
   * @param x Mouse x coordinate
   * @param y Mouse y coordinate
   */
  mouseoverPoint = (x: number, y: number) => {
    // Invert to the stage scale => invert to the data scale
    const dataX = this.xScale.invert(this.curZoomTransform.invertX(x));
    const dataY = this.yScale.invert(this.curZoomTransform.invertY(y));

    // Let the worker to search the closest point in a radius
    const message: TreeWorkerMessage = {
      command: 'startQuadtreeSearch',
      payload: {
        x: dataX,
        y: dataY,
        time: this.timeInspectMode && this.curTime ? this.curTime : '',
        group: ''
      }
    };
    this.treeWorker.postMessage(message);
  };

  /**
   * Event handler for mousemove
   * @param e Mouse event
   */
  mousemoveHandler = (e: MouseEvent) => {
    // Show tooltip when mouse over a data point on canvas
    // We need to use color picking to figure out which point is hovered over
    const x = e.offsetX;
    const y = e.offsetY;
    this.lastMouseClientPosition = { x: x, y: y };

    // Show point highlight
    if (this.showPoint && !this.hideHighlights) {
      this.mouseoverPoint(x, y);
    }

    // Show labels
    if (!this.hideHighlights) {
      this.mouseoverLabel(x, y);
    }
  };

  /**
   * Get the current zoom viewing box
   * @returns Current zoom view box
   */
  getCurZoomBox = () => {
    const box: Rect = {
      x: this.curZoomTransform.invertX(0),
      y: this.curZoomTransform.invertY(0),
      width: Math.abs(
        this.curZoomTransform.invertX(this.svgFullSize.width) -
          this.curZoomTransform.invertX(0)
      ),
      height: Math.abs(
        this.curZoomTransform.invertY(this.svgFullSize.height) -
          this.curZoomTransform.invertY(0)
      )
    };
    return box;
  };

  /**
   * Get the current viewing area's zoom viewing box
   * @returns Current zoom view box
   */
  getCurViewingZoomBox = () => {
    const box: Rect = {
      x: this.curZoomTransform.invertX(config.layout.searchPanelWidth),
      y: this.curZoomTransform.invertY(config.layout.topBarHeight),
      width: Math.abs(
        this.curZoomTransform.invertX(this.svgFullSize.width) -
          this.curZoomTransform.invertX(config.layout.searchPanelWidth)
      ),
      height: Math.abs(
        this.curZoomTransform.invertY(this.svgFullSize.height) -
          this.curZoomTransform.invertY(config.layout.topBarHeight)
      )
    };
    return box;
  };

  /**
   * Get the current view ranges [xmin, xmax, ymin, ymax] in the data coordinate
   * @returns Current view box in the data coordinate
   */
  getCurViewRanges = (): [number, number, number, number] => {
    const zoomBox = this.getCurZoomBox();

    const xMin = this.xScale.invert(zoomBox.x);
    const xMax = this.xScale.invert(zoomBox.x + zoomBox.width);
    const yMin = this.yScale.invert(zoomBox.y + zoomBox.height);
    const yMax = this.yScale.invert(zoomBox.y);

    const result: [number, number, number, number] = [xMin, xMax, yMin, yMax];
    return result;
  };

  /**
   * Handle user changing the display setting
   * @param checkbox Checkbox name
   * @param checked Whether this checkbox is checked
   */
  displayCheckboxChanged = (checkbox: string, checked: boolean) => {
    switch (checkbox) {
      case 'contour': {
        this.showContour = checked;
        this.svg.select('g.contour-group').classed('hidden', !this.showContour);

        if (this.showGrid) {
          let startColor: string;
          let endColor: string;

          if (this.showContour) {
            // No contour -> contour | dark -> light
            startColor = config.gridColorDark;
            endColor = config.gridColorLight;
          } else {
            // Contour -> no contour | light -> dark
            startColor = config.gridColorLight;
            endColor = config.gridColorDark;
          }

          const duration = 300;
          const colorScale = d3.interpolateHsl(startColor, endColor);
          requestAnimationFrame(time => {
            this.drawTopicGridFrame(time, null, duration, colorScale);
          });
        }
        break;
      }

      case 'point': {
        this.showPoint = checked;
        this.pointCanvas
          .classed('hidden', !this.showPoint)
          .classed('faded', this.showPoint && this.showLabel);

        if (this.showPoint) {
          this.drawScatterPlot();
        }

        if (this.showGrid) this.redrawTopicGrid();
        break;
      }

      case 'grid': {
        this.showGrid = checked;
        this.topicCanvases.forEach(c => c.classed('hidden', !this.showGrid));

        if (this.showGrid) {
          this.redrawTopicGrid();
        }

        break;
      }

      case 'label': {
        this.showLabel = checked;
        this.topSvg
          .select('g.top-content g.topics')
          .classed('hidden', !this.showLabel);

        this.pointCanvas.classed('faded', this.showPoint && this.showLabel);

        if (this.showLabel) {
          this.layoutTopicLabels(this.userMaxLabelNum);
        }
        break;
      }

      default: {
        console.error('Unknown checkbox name', checkbox);
        break;
      }
    }
  };
}
