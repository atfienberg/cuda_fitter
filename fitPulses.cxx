#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>

#include "tbb/parallel_for.h"

#include "TFile.h"
#include "TGraph.h"
#include "TTree.h"
#include "TSpline.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include <set>

// header for this little example
#include "header.h"

bool check_results(pulseFinderResultCollection* cpu_results,
                   pulseFinderResultCollection* gpu_results);

void cpu_process(const short* trace, pulseFinderResultCollection* cpu_results,
                 const std::vector<phaseMap>& pmaps,
                 const std::vector<pulseTemplate>& templates);

double evalLinearTemplate(double val, const float* templ, double xMin,
                          double xMax);

double evalPhaseGroupedTemplate(float phase, unsigned int sampleIndex,
                                const float* templ);

typedef struct {
  std::vector<UInt_t>* eventNum;
  std::vector<UInt_t>* caloNum;
  std::vector<UInt_t>* xtalNum;
  std::vector<UInt_t>* islandNum;
  std::vector<Double_t>* pedestal;
  std::vector<Double_t>* energy;
  std::vector<Double_t>* time;
  std::vector<Double_t>* chi2;
} FIT_RESULT;

int main() {
  std::cout << "size of phaseMap " << sizeof(phaseMap) << std::endl;
  std::cout << "size of pulseFitResult " << sizeof(pulseFinderResult)
            << std::endl;

  std::cout << "loading templates..." << std::endl;
  std::vector<phaseMap> pmaps(NSEGMENTS);
  std::vector<pulseTemplate> templates(NSEGMENTS);

  std::unique_ptr<TGraph> phaseTemplate(new TGraph(0));
  phaseTemplate->SetName("phaseTemplate");
  phaseTemplate->SetTitle("phaseTemplate;template index;");
  std::unique_ptr<TGraph> phaseTemplate10(new TGraph(0));
  phaseTemplate10->SetName("phaseTemplate10");
  phaseTemplate10->SetTitle("phaseTemplate10;template index;");
  std::unique_ptr<TGraph> timeTemplate(new TGraph(0));
  timeTemplate->SetName("timeTemplate");
  timeTemplate->SetTitle("timeTemplate;template index;");
  std::unique_ptr<TGraph> timeTemplate10(new TGraph(0));
  timeTemplate10->SetName("timeTemplate10");
  timeTemplate10->SetTitle("timeTemplate10;template index;");

  for (unsigned int i = 0; i < pmaps.size(); ++i) {
    std::unique_ptr<TFile> templFile(new TFile(Form(
        "../art/../art/srcs/gm2calo/g2RunTimeFiles/laserTemplates/"
        "laserTemplateFile%i.root",
        i)));
    std::unique_ptr<TSpline3> phaseMapSpline(
        (TSpline3*)templFile->Get("realTimeSpline"));
    std::unique_ptr<TSpline3> pulseSpline(
        (TSpline3*)templFile->Get("masterSpline"));

    // get time offset between spline max an t = 0
    TF1 splinefunc("splinefunc", [&](double* x, double* p) {
      return pulseSpline->Eval(x[0]);
    }, -20, 20, 0);
    pmaps[i].timeOffset = splinefunc.GetMaximumX();

    for (unsigned int j = 0; j < NPOINTSPHASEMAP; ++j) {
      if (j == 0) {
        pmaps[i].table[j] = 0;
      } else if (j == NPOINTSPHASEMAP - 1) {
        pmaps[i].table[j] = 1;
      } else {
        pmaps[i].table[j] = phaseMapSpline->Eval(j * PHASEMAPSTEP);
      }
    }

    double xmin = -1 * PEAKINDEXINFIT - 0.5 + pmaps[i].timeOffset;
    for (unsigned int phase_i = 0; phase_i <= POINTSPERSAMPLE; ++phase_i) {
      for (unsigned int sample = 0; sample < SAMPLESPERFIT; ++sample) {
        int index = phase_i * POINTSPERSAMPLE + sample;
        double phase = static_cast<double>(phase_i) / POINTSPERSAMPLE;
        templates[i].table[index] = pulseSpline->Eval(xmin + phase + sample);
        if (i == 24) {
          phaseTemplate->SetPoint(phaseTemplate->GetN(), index,
                                  templates[i].table[index]);
          timeTemplate->SetPoint(
              timeTemplate->GetN(), index,
              pulseSpline->Eval(xmin +
                                static_cast<double>(index) / POINTSPERSAMPLE));
          if (phase_i == 10 || phase_i == 11) {
            phaseTemplate10->SetPoint(phaseTemplate10->GetN(), index,
                                      templates[i].table[index]);
            timeTemplate10->SetPoint(
                timeTemplate10->GetN(),
                floor(static_cast<double>(phase + sample) * POINTSPERSAMPLE),
                templates[i].table[index]);
          }
        }
      }
    }
  }
  std::cout << "done loading templates." << std::endl
            << std::endl;

  std::cout << "reading out data from tree..." << std::endl
            << std::endl;
  TFile f(ROOTFILE);
  std::unique_ptr<TTree> tree((TTree*)f.Get("slacAnalyzer/riderTree"));
  std::vector<short>* riderTrace = 0;
  tree->SetBranchAddress("Trace", &riderTrace);
  unsigned int xtalNum = 0;
  unsigned int fillNum = 0;
  tree->SetBranchAddress("XtalNum", &xtalNum);
  tree->SetBranchAddress("EventNum", &fillNum);
  auto host_trace_buf = alloc_pinned<short>(NSEGMENTS * TRACELEN);
  auto gpu_results = alloc_pinned<pulseFinderResultCollection>(NSEGMENTS);
  std::set<unsigned int> xtalNums;
  // read out all the data
  unsigned int nSegsFilled = 0;
  for (unsigned int e = 0; e < tree->GetEntries(); ++e) {
    tree->GetEntry(e);
    if (xtalNum < NSEGMENTS) {
      if (xtalNums.count(xtalNum) != 0) {
        std::cout << "duplicate xtal found: " << xtalNum << std::endl;
        exit(1);
      }
      xtalNums.insert(xtalNum);

      // copy into big trace buffer
      std::copy(riderTrace->begin(), riderTrace->end(),
                host_trace_buf + xtalNum * TRACELEN);
      nSegsFilled++;
      if (nSegsFilled == NSEGMENTS) {
        break;
      }
    }
  }

  // process gpu code
  std::cout << "done loading data from tree." << std::endl;

  std::cout << "begin gpu initializiation!" << std::endl
            << std::endl;
  gpu_init(pmaps, templates);
  std::cout << "gpu init done. starting kernel execution..." << std::endl
            << "------" << std::endl;
  gpu_process(host_trace_buf, gpu_results);

  std::cout << "sorting results by time..." << std::endl;
  for (unsigned int seg = 0; seg < NSEGMENTS; ++seg) {
    std::sort(gpu_results[seg].fit_results,
              gpu_results[seg].fit_results + gpu_results[seg].nPulses,
              [](const auto& l, const auto& r) { return l.time < r.time; });
  }
  std::cout << "done sorting" << std::endl
            << std::endl;

  std::cout << "BEGIN multi threaded cpu verison..." << std::endl;
  pulseFinderResultCollection cpu_results[NSEGMENTS];
  auto start = std::chrono::high_resolution_clock::now();
  cpu_process(host_trace_buf, cpu_results, pmaps, templates);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "time for multithreaded CPU: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl;

  std::cout << "checking results..." << std::endl;
  bool success = check_results(cpu_results, gpu_results);
  if (success) {
    std::cout << "gpu results match cpu results." << std::endl;
  }
  std::cout << "try repeating..." << std::endl;

  gpu_process(host_trace_buf, gpu_results);
  // sort results
  for (unsigned int seg = 0; seg < NSEGMENTS; ++seg) {
    std::sort(gpu_results[seg].fit_results,
              gpu_results[seg].fit_results + gpu_results[seg].nPulses,
              [](const auto& l, const auto& r) { return l.time < r.time; });
  }

  std::cout << "checking second pass..." << std::endl;
  success = check_results(cpu_results, gpu_results);
  if (success) {
    std::cout << "second pass gpu results match cpu results." << std::endl;
  }

  std::cout << "n pulses: ";
  std::cout << gpu_results[0].nPulses;
  for (unsigned int i = 1; i < NSEGMENTS; ++i) {
    std::cout << ", " << gpu_results[i].nPulses;
  }
  std::cout << std::endl;
  tree.reset((TTree*)f.Get("slacAnalyzer/eventTree"));

  // test against FitResultArtRecords
  std::cout << std::endl
            << "checking against fit result art records..." << std::endl;

  FIT_RESULT fitresult = {};
  tree->SetBranchAddress("FitResult_EventNum", &fitresult.eventNum);
  tree->SetBranchAddress("FitResult_CaloNum", &fitresult.caloNum);
  tree->SetBranchAddress("FitResult_XtalNum", &fitresult.xtalNum);
  tree->SetBranchAddress("FitResult_IslandNum", &fitresult.islandNum);
  tree->SetBranchAddress("FitResult_Pedestal", &fitresult.pedestal);
  tree->SetBranchAddress("FitResult_Energy", &fitresult.energy);
  tree->SetBranchAddress("FitResult_Time", &fitresult.time);
  tree->SetBranchAddress("FitResult_Chi2", &fitresult.chi2);
  tree->GetEntry(fillNum - 1);
  pulseFinderResultCollection frResults[NSEGMENTS];
  pulseFinderResultCollection empty = {};
  std::fill(frResults, frResults + NSEGMENTS, empty);
  for (unsigned int i = 0; i < fitresult.energy->size(); ++i) {
    unsigned int xtalIndex = fitresult.xtalNum->at(i);
    // std::cout << "xtal num " << xtalIndex << std::endl;
    // double fre = fitresult.energy->at(i);
    // double gpe = gpu_results[xtalIndex].fit_results[0].energy;
    if (fitresult.chi2->at(i) > 0 && fitresult.energy->at(i) > 2000) {
      unsigned int pulseNum = frResults[xtalIndex].nPulses++;
      frResults[xtalIndex].fit_results[pulseNum].energy =
          fitresult.energy->at(i);
      frResults[xtalIndex].fit_results[pulseNum].time = fitresult.time->at(i);
    }
  }

  std::cout << "fr n pulses: ";
  std::cout << frResults[0].nPulses;
  for (unsigned int i = 1; i < NSEGMENTS; ++i) {
    std::cout << ", " << frResults[i].nPulses;
  }
  std::cout << std::endl;

  std::unique_ptr<TFile> outf(new TFile("gpuVsArtPlots.root", "recreate"));
  std::vector<std::unique_ptr<TGraph>> frgraphs;
  std::vector<std::unique_ptr<TGraph>> gpugraphs;
  std::unique_ptr<TH1D> phaseHist(
      new TH1D("phaseHist", "phaseHist", 100, 0, 1));
  std::unique_ptr<TH1D> timeDiffHist(
      new TH1D("timeDiffHist", "t_{gpu} - t_{offline}", 1000, 0.0, 0.0));
  std::unique_ptr<TH1D> energyDiffHist(new TH1D(
      "energyDiffHist", "energy_{gpu} - energy_{offline}", 100, 0.0, 0.0));
  std::unique_ptr<TH2D> energyCorrelation(
      new TH2D("energyCorrelation", ";energy_{offline};energy_{gpu}", 100, 0.0,
               0.0, 100, 0.0, 0.0));
  std::unique_ptr<TH2D> timeCorrelation(
      new TH2D("timeCorrelation",
               ";time_{offline} - peak sample;time_{gpu} - peak sample", 100,
               0.0, 0.0, 100, 0.0, 0.0));
  for (unsigned int i = 0; i < NSEGMENTS; ++i) {
    frgraphs.emplace_back(new TGraph(0));
    TGraph* frg = frgraphs.back().get();
    frg->SetName(Form("xtal%iArtGraph", i));
    frg->SetTitle(
        "offline art fitter results;fit time mod 1000 [clock ticks]; fitted "
        "pulse area");
    for (unsigned int p_num = 0; p_num < frResults[i].nPulses; ++p_num) {
      const auto& result = frResults[i].fit_results[p_num];
      if (result.time < 1000 || result.time > 3000) {
        frg->SetPoint(frg->GetN(), fmod(result.time, 1000), result.energy);
      }
    }

    gpugraphs.emplace_back(new TGraph(0));
    TGraph* gpug = gpugraphs.back().get();
    gpug->SetName(Form("xtal%iGpuGraph", i));
    gpug->SetTitle(
        "gpu fitter results;fit time mod 1000 [clock ticks]; fitted pulse "
        "area");
    for (unsigned int p_num = 0; p_num < gpu_results[i].nPulses; ++p_num) {
      const auto& result = gpu_results[i].fit_results[p_num];
      if (result.time < 1000 || result.time > 3000) {
        gpug->SetPoint(gpug->GetN(), fmod(result.time, 1000), result.energy);
        phaseHist->Fill(result.phase);

        auto beginIter = frResults[i].fit_results;
        auto endIter = frResults[i].fit_results + frResults[i].nPulses;
        auto matchingFRIter =
            std::find_if(beginIter, endIter, [&](const pulseFinderResult& r) {
              return std::abs(r.time - result.time) < 1;
            });
        if (matchingFRIter != endIter) {
          timeDiffHist->Fill(
              (result.peak_index + 0.5 - result.phase - pmaps[i].timeOffset) -
              matchingFRIter->time);
          energyDiffHist->Fill(result.energy - matchingFRIter->energy);
          energyCorrelation->Fill(matchingFRIter->energy, result.energy);
          timeCorrelation->Fill(matchingFRIter->time - result.peak_index,
                                result.peak_index + 0.5 - result.phase -
                                    pmaps[i].timeOffset - result.peak_index);
        }
      }
    }

    frg->SetMarkerColor(kRed);
    frg->Write();
    gpug->SetMarkerColor(kBlue);
    gpug->Write();
  }

  phaseHist->Write();
  timeDiffHist->Write();
  energyDiffHist->Write();
  phaseTemplate->Write();
  timeTemplate->Write();
  phaseTemplate10->Write();
  timeTemplate10->Write();
  energyCorrelation->Write();
  timeCorrelation->Write();

  std::vector<char> bankbuff(sizeof(pulseFinderResultCollection) * NSEGMENTS +
                             sizeof(unsigned int) * NSEGMENTS);
  std::cout
      << "testing time to flatten fit results into midas bank like array: "
      << std::endl;
  start = std::chrono::high_resolution_clock::now();
  // write in the number of segments
  unsigned int* nSegsP = (unsigned int*)bankbuff.data();
  *nSegsP = NSEGMENTS;
  char* datap = bankbuff.data() + sizeof(unsigned int);
  for (pulseFinderResultCollection* iter = gpu_results;
       iter != gpu_results + NSEGMENTS; ++iter) {
    // pack in nPulses
    unsigned int* dataPuint = (unsigned int*)datap;
    *dataPuint = iter->nPulses;

    pulseFinderResult* dataPfit_result = (pulseFinderResult*)(dataPuint + 1);
    datap = (char*)std::copy(
        iter->fit_results, iter->fit_results + iter->nPulses, dataPfit_result);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "time for bank packing: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl;
  std::cout << "bank size: " << std::distance(bankbuff.data(), datap) / 1000.0
            << " kilobytes" << std::endl;

  std::cout << "testing time to unpack midas bank like array: " << std::endl;
  start = std::chrono::high_resolution_clock::now();
  unsigned int nsegsUnpacked = *(unsigned int*)bankbuff.data();
  typedef std::vector<pulseFinderResult> fitResultCollection;
  std::vector<fitResultCollection> unpackerResults(nsegsUnpacked);

  datap = (char*)(bankbuff.data() + sizeof(unsigned int));
  for (unsigned int i = 0; i < nsegsUnpacked; ++i) {
    unsigned int* nPulsesp = (unsigned int*)datap;
    auto npulses = *nPulsesp;

    unpackerResults[i].resize(npulses);
    pulseFinderResult* dataPfresults = (pulseFinderResult*)(nPulsesp + 1);
    std::copy(dataPfresults, dataPfresults + npulses,
              unpackerResults[i].begin());
    datap = (char*)(dataPfresults + npulses);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "time for bank unpacking: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start).count() << " microseconds" << std::endl << std::endl;
                   
  std::cout << "test a pulse for unpacking integrity: seg 15 pulse 32" << std::endl;
  const auto& r = unpackerResults[15][32];
  std::cout << "energy: " << r.energy << std::endl;
  std::cout << "time: " << r.time << std::endl;
  std::cout << "phase: " << r.phase << std::endl;
  std::cout << "chi2: " << r.chi2 << std::endl;

  std::cout << "closing gpu" << std::endl;
  free_pinned(host_trace_buf);
  free_pinned(gpu_results);
  gpu_close();
}

constexpr double tolerance_permille = 0.1;

bool eq_within_tolerance(float a, float b) {
  float avg = (a + b) / 2.0;
  if (avg == 0) {
    return true;
  }
  return std::abs(a - b) / avg < (tolerance_permille / 1000.0);
}

void cpu_process(const short* trace, pulseFinderResultCollection* cpu_results,
                 const std::vector<phaseMap>& pmaps,
                 const std::vector<pulseTemplate>& templates) {
  pulseFinderResultCollection emptyRes = {};
  // zero it
  std::fill(cpu_results, cpu_results + NSEGMENTS, emptyRes);
  tbb::parallel_for(
      std::size_t(0), std::size_t(NSEGMENTS), [&](std::size_t seg) {
        cpu_results[seg].nPulses = 0;
        for (unsigned int i = MINFITTIME;
             i < TRACELEN - (SAMPLESPERFIT - PEAKINDEXINFIT - 1); ++i) {
          short l, m, r;
          l = trace[seg * TRACELEN + i - 1];
          m = trace[seg * TRACELEN + i];
          r = trace[seg * TRACELEN + i + 1];
          if ((m < THRESHOLD) && ((m <= l) && (m < r))) {
            //            found pulse, fit
            const short* subtrace = &trace[seg * TRACELEN + i - PEAKINDEXINFIT];
            const short* min_iter = &trace[seg * TRACELEN + i];

            float ptime;
            if (min_iter[0] == min_iter[1]) {
              ptime = 1;
            } else {
              ptime = 2.0 / M_PI *
                      atan(static_cast<float>(min_iter[-1] - min_iter[0]) /
                           (min_iter[1] - min_iter[0]));
            }
            float rt = evalLinearTemplate(ptime, &pmaps[seg].table[0], 0, 1);
            float time = i + rt - 0.5 - pmaps[seg].timeOffset;
            float phase = 1 - rt;
            float dDotT = 0;
            float T2 = 0;
            float dSum = 0;
            float tSum = 0;
            float ti[SAMPLESPERFIT];
            float di[SAMPLESPERFIT];
            for (unsigned int samp = 0; samp < SAMPLESPERFIT; ++samp) {
              ti[samp] = evalPhaseGroupedTemplate(phase, samp,
                                                  &templates[seg].table[0]);
              di[samp] = subtrace[samp];
              tSum += ti[samp];
              dSum += di[samp];
              dDotT += ti[samp] * di[samp];
              T2 += ti[samp] * ti[samp];
            }
            auto n = SAMPLESPERFIT;
            float denomRecip = 1.0 / (tSum * tSum - n * T2);
            float s = denomRecip * (dSum * tSum - n * dDotT);
            float p = denomRecip * (dDotT * tSum - dSum * T2);
            float chi2 = 0;
            for (unsigned int samp = 0; samp < SAMPLESPERFIT; ++samp) {
              float resid_i = di[samp] - s * ti[samp] - p;
              chi2 += resid_i * resid_i;
            }
            if (s < 0) {
              s = -1 * s;
            }

            auto i_res = cpu_results[seg].nPulses;
            // // record results
            if (i_res < OUTPUTARRAYLEN) {
              cpu_results[seg].nPulses += 1;
              cpu_results[seg].fit_results[i_res].time = time;
              cpu_results[seg].fit_results[i_res].phase = phase;
              cpu_results[seg].fit_results[i_res].energy = s;
              cpu_results[seg].fit_results[i_res].pedestal = p;
              cpu_results[seg].fit_results[i_res].chi2 = chi2;
              cpu_results[seg].fit_results[i_res].peak_index = i;
              cpu_results[seg].fit_results[i_res].peak_value = m;
            }
          }
        }
      });
}

bool check_results(pulseFinderResultCollection* cpu_results,
                   pulseFinderResultCollection* gpu_results) {
  bool success = true;
  // int failureXtal = 0;
  for (unsigned int seg = 0; success && seg < NSEGMENTS; ++seg) {
    success &= gpu_results[seg].nPulses == cpu_results[seg].nPulses;
    if (!success) {
      std::cout << "failure!" << std::endl;
      std::cout << "xtal " << seg << " n pulses: CPU "
                << cpu_results[seg].nPulses << ", GPU "
                << gpu_results[seg].nPulses << std::endl;
      break;
    }
    for (unsigned int j = 0; j < gpu_results[seg].nPulses; ++j) {
      const auto& cfr = cpu_results[seg].fit_results[j];
      const auto& gfr = gpu_results[seg].fit_results[j];
      success &= eq_within_tolerance(cfr.energy, gfr.energy);
      success &= eq_within_tolerance(cfr.pedestal, gfr.pedestal);
      success &= eq_within_tolerance(cfr.chi2, gfr.chi2);
      success &= eq_within_tolerance(cfr.time, gfr.time);
      success &= eq_within_tolerance(cfr.phase, gfr.phase);
      if (!success) {
        std::cout << "mismatch, seg " << seg << " pulse " << j << std::endl;
        std::cout << "cpu energy: " << cfr.energy
                  << ", gpu energy: " << gfr.energy << std::endl;
        std::cout << "cpu phase: " << cfr.phase << ", gpu phase: " << gfr.phase
                  << std::endl;
        std::cout << "cpu time: " << cfr.time << ", gpu time: " << gfr.time
                  << std::endl;
        std::cout << "cpu pedestal: " << cfr.pedestal
                  << ", gpu pedestal: " << gfr.pedestal << std::endl;
        std::cout << "cpu chi2: " << cfr.chi2 << ", gpu chi2: " << gfr.chi2
                  << std::endl;
        std::cin.ignore();
        break;
      }
    }
  }
  return success;
}

double evalLinearTemplate(double val, const float* templ, double xMin,
                          double xMax) {
  //  double stepsPerTime = (1024 - 1) / (xMax - xMin);
  double where = (val - xMin) * (NPOINTSPHASEMAP - 1) / (xMax - xMin);
  int low = std::floor(where);
  double dt = where - low;
  return templ[low] * (1 - dt) + templ[low + 1] * dt;
}

double evalPhaseGroupedTemplate(float phase, unsigned int sampleIndex,
                                const float* templ) {
  float phase_loc = phase * POINTSPERSAMPLE;
  int phase_index = std::floor(phase_loc);
  double weight_high = phase_loc - phase_index;
  if (phase_index < 0) {
    phase_index = 0;
    weight_high = 0;
  } else if (phase_index >= static_cast<int>(POINTSPERSAMPLE)) {
    phase_index = POINTSPERSAMPLE - 1;
    weight_high = 1;
  }
  int low_index = phase_index * SAMPLESPERFIT + sampleIndex;
  float low_point = templ[low_index];
  int high_index = low_index + SAMPLESPERFIT;
  float high_point = templ[high_index];
  return low_point * (1 - weight_high) + high_point * weight_high;
}
