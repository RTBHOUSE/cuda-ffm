#include "training_session.h"
#include "ffm_trainer.h"

void learn(const int argc, const char** argv)
{
    const Options options = parseOptions(argc, argv);

    TrainingSession session(options);
    session.trainModel();
    session.exportModel();
}

int main(int const argc, const char ** argv)
{
    FFMStatic::init();
    learn(argc, argv);
    FFMStatic::destroy();
    return EXIT_SUCCESS;
}
