#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include "math_constants.h"
#include "header.h"

constexpr int PULSESPERBLOCK = 4;
constexpr int FIT_THREADSPERBLOCK = PULSESPERBLOCK * SAMPLESPERFIT;

__device__ phaseMap d_phase_maps[NSEGMENTS];
__device__ pulseTemplate d_templates[NSEGMENTS];

__global__ void find_times(const short* trace,
                           pulseFinderResultCollection* resultcol, short threshold, short polarity) {
  for (uint segment_num = 0; segment_num < NSEGMENTS; ++segment_num) {
    // find index based on global id
    uint sample_num = blockIdx.x * blockDim.x + threadIdx.x;
    uint trace_index = segment_num * TRACELEN + sample_num;

    // don't continue if your trace_index is out of bounds for pulse fitting
    if (sample_num < MINFITTIME ||
        sample_num >= TRACELEN - (SAMPLESPERFIT - PEAKINDEXINFIT - 1)) {
      return;
    }

    // we need the samples at this point and surrounding (left, middle, right)
    short m = polarity * trace[trace_index];
    // if this sample is a local minimum and is over threshold, record it
    if (m > polarity * threshold) {
      short l = polarity * trace[trace_index - 1];
      short r = polarity * trace[trace_index + 1];

      // must be local minimum, but since we have digital ADC values
      // we must allow for max to equal sample on left, but not on right
      // if we allow max sample to equal right, we will fit same pulse twice
      if ((m >= l) && (m > r)) {
        uint pulse_index = atomicAdd(&resultcol[segment_num].nPulses, 1);
        if (pulse_index < OUTPUTARRAYLEN) {
          // find pulse time, phase
          // first calculate pseudo time
          float numerator = l - m;
          float denominator = r - m;
          // denominator can't be zero because m < r
          float ptime = 2.0 / CUDART_PI_F * atan(numerator / denominator);

          // next interpolate time map table
          float where = ptime * (NPOINTSPHASEMAP - 1);
          int low_index = floor(where);
          float weight_heigh = where - low_index;
          float real_time;
          // check for out of bounds
          if (low_index < 0) {
            real_time = 0;
          } else if (low_index >= NPOINTSPHASEMAP - 1) {
            real_time = 1.0;
          } else {
            // do the interpolation
            real_time =
                d_phase_maps[segment_num].table[low_index] *
                    (1 - weight_heigh) +
                d_phase_maps[segment_num].table[low_index + 1] * weight_heigh;
          }

          // record time, phase, peak index, peak value
          float time_offset = d_phase_maps[segment_num].timeOffset;
          resultcol[segment_num].fit_results[pulse_index].time =
              sample_num + real_time - 0.5 - time_offset;
          resultcol[segment_num].fit_results[pulse_index].phase = 1 - real_time;
          resultcol[segment_num].fit_results[pulse_index].peak_index =
              sample_num;
          resultcol[segment_num].fit_results[pulse_index].peak_value = m;
        } else {
          // beyond limit for number of pulses we're trying to fit
          atomicSub(&resultcol[segment_num].nPulses, 1);
        }
      }
    }
  }
}

// according to info on nvidia.com, no need to explicitly synchronize threads
// within groups of 32
// because warps are 32 threads and instructions in warp are always synchronous
__global__ void fit_pulses(const short* trace,
                           pulseFinderResultCollection* resultcol) {
  // arrays for accumulation
  __shared__ float tSum[FIT_THREADSPERBLOCK];
  __shared__ float dSum[FIT_THREADSPERBLOCK];
  __shared__ float dDotT[FIT_THREADSPERBLOCK];
  __shared__ float tDotT[FIT_THREADSPERBLOCK];

  unsigned int segment_num = blockIdx.y;
  unsigned int pulse_num =
      blockIdx.x * PULSESPERBLOCK + threadIdx.x / SAMPLESPERFIT;

  // return asap if this pulse doesn't exit
  if ((pulse_num >= OUTPUTARRAYLEN) ||
      (pulse_num >= resultcol[segment_num].nPulses)) {
    return;
  }
  // step one: read needed inputs from resultcol
  float phase = resultcol[segment_num].fit_results[pulse_num].phase;
  unsigned int start_sample =
      segment_num * TRACELEN +
      resultcol[segment_num].fit_results[pulse_num].peak_index - PEAKINDEXINFIT;

  // step two: read in template values for this phase and sample num
  unsigned int sample_index = threadIdx.x % SAMPLESPERFIT;
  float phase_loc = phase * POINTSPERSAMPLE;
  int phase_index = floor(phase_loc);
  float weight_high = phase_loc - phase_index;
  // make sure we're in bounds
  if (phase_index < 0) {
    phase_index = 0;
    weight_high = 0;
  } else if (phase_index >= POINTSPERSAMPLE) {
    phase_index = POINTSPERSAMPLE - 1;
    weight_high = 1;
  }
  unsigned int low_index = phase_index * SAMPLESPERFIT + sample_index;
  unsigned int high_index = low_index + SAMPLESPERFIT;
  float low_value = d_templates[segment_num].table[low_index];
  float high_value = d_templates[segment_num].table[high_index];

  // step 2.5 evaluate template
  float t_i = low_value * (1 - weight_high) + high_value * weight_high;

  // step three : read in pulse value
  float d_i = trace[start_sample + sample_index];

  // step four: prepare accumulation/reduction arrays
  tSum[threadIdx.x] = t_i;
  dSum[threadIdx.x] = d_i;
  dDotT[threadIdx.x] = d_i * t_i;
  tDotT[threadIdx.x] = t_i * t_i;

  // step five: accumulate, note that explicit synchronization
  // is not required because all accumulation is done within a warp
  // it seems like this stops working if the if and for are inverted, so don't
  // do that
  for (unsigned int stride = 16; stride >= 1; stride /= 2) {
    if (sample_index < 16) {
      tSum[threadIdx.x] += tSum[threadIdx.x + stride];
      dSum[threadIdx.x] += dSum[threadIdx.x + stride];
      dDotT[threadIdx.x] += dDotT[threadIdx.x + stride];
      tDotT[threadIdx.x] += tDotT[threadIdx.x + stride];
    }
  }

  // step six : calculate pedestal, energy
  // read final accumulated results
  int result_index = (threadIdx.x / SAMPLESPERFIT) * SAMPLESPERFIT;
  float tSumFinal = tSum[result_index];
  float dSumFinal = dSum[result_index];
  float dDotTFinal = dDotT[result_index];
  float tDotTFinal = tDotT[result_index];

  float denomRecip = 1.0 / (tSumFinal * tSumFinal - SAMPLESPERFIT * tDotTFinal);

  float energy =
      denomRecip * (dSumFinal * tSumFinal - SAMPLESPERFIT * dDotTFinal);
  float pedestal =
      denomRecip * (dDotTFinal * tSumFinal - dSumFinal * tDotTFinal);

  // step seven: load partial chi^2s into shared memory
  __shared__ float chi2sum[FIT_THREADSPERBLOCK];
  float residual_i = d_i - energy * t_i - pedestal;
  chi2sum[threadIdx.x] = residual_i * residual_i;

  // step eight: accumulate partial chi2s
  for (unsigned int stride = 16; stride >= 1; stride /= 2) {
    if (sample_index < 16) {
      chi2sum[threadIdx.x] += chi2sum[threadIdx.x + stride];
    }
  }

  // final step: record results

  // force energy positive
  if (energy < 0) {
    energy = energy * -1;
  }
  if (sample_index == 0) {
    resultcol[segment_num].fit_results[pulse_num].energy = energy;
    resultcol[segment_num].fit_results[pulse_num].pedestal = pedestal;
    resultcol[segment_num].fit_results[pulse_num].chi2 = chi2sum[threadIdx.x];
  }
}

// some global variables to keep track of state easily
short* device_trace = nullptr;
pulseFinderResultCollection* device_result = nullptr;
cudaDeviceProp prop;

// set device and malloc buffers
void gpu_init(std::vector<phaseMap>& phaseMaps,
              std::vector<pulseTemplate>& pulseTemplates) {
  int count;
  cudaGetDeviceCount(&count);
  std::cout << "there are " << count << " devices available." << std::endl;

  cudaSetDevice(0);
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "using gpu " << prop.name << std::endl;

  cudaMalloc((void**)&device_trace, trace_size);
  cudaMalloc((void**)&device_result, result_size);

  cudaMemcpyToSymbol(d_phase_maps, (void*)phaseMaps.data(), phase_maps_size);
  cudaMemcpyToSymbol(d_templates, (void*)pulseTemplates.data(), templates_size);
}

// free buffers, close gpu
void gpu_close() {
  if (device_trace) {
    std::cout << "freeing device trace" << std::endl;
    cudaFree(device_trace);
  }
  if (device_result) {
    std::cout << "freeing device result" << std::endl;
    cudaFree(device_result);
  }
  cudaDeviceReset();
}

void gpu_process(short* host_trace, pulseFinderResultCollection* result_buff) {
  // copy data to gpu
  auto fullGpuStart = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(device_trace, host_trace, trace_size, cudaMemcpyHostToDevice);

  // zero results
  cudaMemset(device_result, 0, result_size);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "time for write to GPU: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl
            << std::endl;

  // launch job
  // 128 seems to give best performance
  int threadsPerBlock = 128;
  dim3 dimBlock(threadsPerBlock);
  std::cout << "threads per block: " << threadsPerBlock << std::endl;
  int blocks_per_trace =
      std::ceil(static_cast<double>(TRACELEN) / threadsPerBlock);
  std::cout << "blocks per trace: " << blocks_per_trace << std::endl;
  std::cout << "threads per trace: " << blocks_per_trace* threadsPerBlock
            << std::endl;
  dim3 dimGrid(blocks_per_trace);
  start = std::chrono::high_resolution_clock::now();
  find_times<<<dimGrid, dimBlock>>> (device_trace, device_result, THRESHOLD, -1);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "time for find pulses computation on GPU : "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl
            << std::endl;

  std::cout << "TRYING FIT" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  dimGrid = dim3(OUTPUTARRAYLEN / PULSESPERBLOCK, NSEGMENTS);
  dimBlock = dim3(SAMPLESPERFIT * PULSESPERBLOCK, 1);
  fit_pulses<<<dimGrid, dimBlock>>> (device_trace, device_result);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "time for fit pulses computation on GPU : "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl
            << std::endl;
  // get data from gpu, finish up
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpy((void*)result_buff, device_result, result_size,
             cudaMemcpyDeviceToHost);
  end = std::chrono::high_resolution_clock::now();
  auto fullGpuEnd = end;
  std::cout << "time read out results from GPU: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl
            << std::endl;
  std::cout << "time from before write to gpu to after read from gpu: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   fullGpuEnd - fullGpuStart).count() << " microseconds"
            << std::endl
            << std::endl;
}
