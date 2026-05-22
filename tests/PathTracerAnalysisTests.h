#ifndef LAPHRIAENGINE_PATHTRACERANALYSISTESTS_H
#define LAPHRIAENGINE_PATHTRACERANALYSISTESTS_H

bool testPathTracerBaselineSweepMatrix();
bool testPathTracerPercentiles();
bool testPathTracerScoreBudgetGate();
bool testPathTracerDebugAovContract();
bool testPathTracerRemovedGiCacheGuard();
bool testPathTracerHistoryClampPreservesDimIndirectHistory();
bool testPathTracerPowerHeuristic();

#endif
