// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace protomotions::deployment {

// A bounded, latest-only handoff for asynchronous SDK ingress. The handoff
// never waits for the controller and the consumer never drains an old queue:
// it takes the newest complete sample while keeping the policy/runtime
// contract explicit.
//
// Ownership is shared so an SDK adapter can publish a snapshot without a lock
// in the control thread. Producers publish immutable snapshots with a
// positive sequence number; stale or replayed sequences are rejected by the
// mailbox itself. `publish` may allocate and returns false on allocation
// failure, so a hard-real-time callback should use a preallocated/non-real-
// time ingress adapter and handle that result explicitly. This is the default
// handoff policy for asynchronous adapter ingress; an adapter that truly needs
// every frame must opt out explicitly and reject skipped sequence numbers.
template <typename T>
class LatestMailbox final {
 public:
  struct Snapshot {
    std::shared_ptr<const T> value;
    std::uint64_t sequence = 0;
  };

  LatestMailbox() = default;
  LatestMailbox(const LatestMailbox&) = delete;
  LatestMailbox& operator=(const LatestMailbox&) = delete;

  bool publish(std::shared_ptr<const T> value, std::uint64_t sequence) noexcept {
    if (!value || sequence == 0) return false;
    try {
      auto entry =
          std::make_shared<const Entry>(Entry{std::move(value), sequence});
      // Keep the replay barrier separate from the visible immutable entry.
      // Producers may advance the barrier before publishing the entry, but
      // consumers only read entry_, so they can never observe a sequence
      // paired with a partially-written snapshot. The barrier's purpose is
      // solely to reject delayed callbacks after clear() or a competing
      // publish; snapshot() is the atomic consumer view.
      auto previous_sequence =
          last_sequence_.load(std::memory_order_acquire);
      while (true) {
        if (sequence <= previous_sequence) return false;
        if (last_sequence_.compare_exchange_weak(
                previous_sequence, sequence, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          break;
        }
      }
      auto current = std::atomic_load_explicit(&entry_, std::memory_order_acquire);
      while (true) {
        if (current && sequence <= current->sequence) return false;
        if (std::atomic_compare_exchange_weak_explicit(
                &entry_, &current, entry, std::memory_order_release,
                std::memory_order_acquire)) {
          return true;
        }
      }
    } catch (...) {
      return false;
    }
  }

  Snapshot snapshot() const noexcept {
    const auto entry =
        std::atomic_load_explicit(&entry_, std::memory_order_acquire);
    return entry ? Snapshot{entry->value, entry->sequence} : Snapshot{};
  }

  std::shared_ptr<const T> latest() const noexcept {
    return snapshot().value;
  }

  std::uint64_t latest_sequence() const noexcept {
    return snapshot().sequence;
  }

  // Clear the visible snapshot without clearing the sequence barrier. This
  // prevents a delayed pre-clear callback from being accepted after a reset.
  void clear() noexcept {
    std::atomic_store_explicit(&entry_, std::shared_ptr<const Entry>{},
                               std::memory_order_release);
  }

 private:
  struct Entry {
    std::shared_ptr<const T> value;
    std::uint64_t sequence;
  };

  std::shared_ptr<const Entry> entry_;
  // Clearing the visible snapshot must not reset the replay barrier. A
  // delayed SDK callback remains stale even while the mailbox is empty.
  std::atomic<std::uint64_t> last_sequence_{0};
};

}  // namespace protomotions::deployment
