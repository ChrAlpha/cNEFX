#pragma once

#include "cnefx/core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace cnefx::detail {
// Zero jobs selects the hardware concurrency. Never exceeds the task count.
inline unsigned concurrency(unsigned jobs, unsigned tasks) {
  const unsigned limit = jobs ? jobs : std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, std::min(limit, tasks));
}

// Runs task(index, poll) for every index in [0, count). The caller's thread
// takes part and is the only thread that runs check, so callbacks keep their
// caller-thread contract; it keeps polling check while other threads finish.
// Other threads poll a stop flag instead. After a task or check fails, the
// remaining threads stop at their next poll and the first failure is rethrown
// once every thread has joined. Failing to start a thread only reduces
// parallelism.
template <class Task>
void parallel_for(unsigned count, unsigned jobs, const Check &check, const Task &task) {
  const auto threads = concurrency(jobs, count);
  if (threads == 1) {
    for (unsigned i = 0; i < count; ++i) {
      if (check)
        check();
      task(i, check);
    }
    return;
  }
  struct Stopped {};
  std::atomic<unsigned> next{0};
  std::atomic<bool> stop{false};
  std::mutex mutex;
  std::exception_ptr failure;
  const auto fail = [&] {
    const std::lock_guard lock(mutex);
    if (!failure)
      failure = std::current_exception();
    stop = true;
  };
  const Check stopped = [&] {
    if (stop.load(std::memory_order_relaxed))
      throw Stopped{};
  };
  const Check caller = [&] {
    stopped();
    if (check)
      check();
  };
  const auto run = [&](const Check &poll) {
    for (unsigned i = 0; !stop && (i = next++) < count;) {
      try {
        poll();
        task(i, poll);
      } catch (const Stopped &) {
      } catch (...) {
        fail();
      }
    }
  };
  std::condition_variable finished;
  unsigned active = 0;
  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  for (unsigned t = 1; t < threads; ++t) {
    {
      const std::lock_guard lock(mutex);
      ++active;
    }
    try {
      workers.emplace_back([&] {
        run(stopped);
        const std::lock_guard lock(mutex);
        --active;
        finished.notify_one();
      });
    } catch (...) {
      const std::lock_guard lock(mutex);
      --active;
      break;
    }
  }
  run(caller);
  std::unique_lock lock(mutex);
  while (!finished.wait_for(lock, std::chrono::milliseconds(20), [&] { return !active; })) {
    lock.unlock();
    try {
      if (!stop && check)
        check();
    } catch (...) {
      fail();
    }
    lock.lock();
  }
  lock.unlock();
  for (auto &worker : workers)
    worker.join();
  if (failure)
    std::rethrow_exception(failure);
}
} // namespace cnefx::detail
