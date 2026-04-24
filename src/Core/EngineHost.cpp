#include "EngineHost.h"

#include "EngineCore.h"

EngineHost::EngineHost() = default;

EngineHost::EngineHost(EngineHostOptions optionsIn, EngineHostCallbacks callbacksIn)
    : options(std::move(optionsIn)), callbacks(std::move(callbacksIn))
{
}

void EngineHost::run() const {
	EngineCore core(options, callbacks);
	core.run();
}
