#ifndef cudapulsefittertestheader
#define cudapulsefittertestheader

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// constexpr char ROOTFILE[] = "../../opencl/gm2slac_run02061.root";
constexpr char ROOTFILE[] = "gm2slac_run03357.root";
constexpr std::size_t TRACELEN = 560000;
constexpr std::size_t NSEGMENTS = 54;
constexpr std::size_t OUTPUTARRAYLEN = 400;
constexpr short THRESHOLD = 1500;

typedef struct {
  float time;
  float phase;
  float energy;
  float pedestal;
  float chi2;
  unsigned int peak_index;
  unsigned int peak_value;
} pulseFinderResult;

typedef struct {
  unsigned int nPulses;
  pulseFinderResult fit_results[OUTPUTARRAYLEN];
} pulseFinderResultCollection;

// phasemap table will be arranged in natural order of time,
// i.e. table_i = phaseMap(i * stepSize)
// where stepSize = 1.0 / (NPOINTSPHASEMAP - 1)
// time offset accounts for difference between pulse template max and t0
constexpr std::size_t NPOINTSPHASEMAP = 32 * 32;
constexpr double PHASEMAPSTEP = 1.0 / (NPOINTSPHASEMAP - 1);
typedef struct {
  float table[NPOINTSPHASEMAP];
  float timeOffset;
} phaseMap;
constexpr std::size_t phase_maps_size = sizeof(phaseMap) * NSEGMENTS;

// pulse templates are arranged in weird way,
// grouped by phase rather than time, such that
// templ_ph,i = templ[ph*NPHASES + i] = T(xmin + phase  + i)
// x_min = -PEAKINDEXINFIT - 0.5 + timeOffset
// time offset is from the phaseMaps and phase is found using the phase map
// this arrangement and definition allows for threads in GPU to only need access
// to contiguous memory areas
constexpr int PEAKINDEXINFIT = 9;
constexpr int MINFITTIME = PEAKINDEXINFIT;
// !warning, don't change samples per fit, it should be 32 to match nvidia warp
// size
constexpr std::size_t SAMPLESPERFIT = 32;
constexpr std::size_t POINTSPERSAMPLE = 32;
// need POINTSPERSAMPLE+1 to accomodate pulses with phase in the highest slot
constexpr std::size_t NPOINTSTEMPLATE = (POINTSPERSAMPLE + 1) * SAMPLESPERFIT;
typedef struct { float table[NPOINTSTEMPLATE]; } pulseTemplate;
constexpr std::size_t templates_size = sizeof(pulseTemplate) * NSEGMENTS;

constexpr std::size_t result_size =
    sizeof(pulseFinderResultCollection) * NSEGMENTS;
constexpr std::size_t trace_size = sizeof(short) * NSEGMENTS * TRACELEN;

template <typename T>
T* alloc_pinned(std::size_t n_elements) {
  T* pointer;
  if (cudaMallocHost((void**)&pointer, n_elements * sizeof(T)) != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMallocHost failed"));
  }
  return pointer;
}

template <typename T>
void free_pinned(T* pointer) {
  if (cudaFreeHost((void*)pointer) != cudaSuccess) {
    throw std::runtime_error(std::string("cudaFreeHost failed"));
  }
}

void gpu_init(std::vector<phaseMap>& phaseMaps,
              std::vector<pulseTemplate>& pulseTemplates);
void gpu_process(short* host_trace, pulseFinderResultCollection* result_buff);
void gpu_close();

#endif
