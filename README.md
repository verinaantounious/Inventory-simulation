

# Appliance Store Inventory Simulation

**A stochastic simulation project for Supply Chain Optimization.**

## 📌 Project Overview

This project simulates an electric home appliance store managing two products (A and B) with shared storage constraints. It uses **Monte Carlo simulation** to model daily demand and lead time uncertainty.

### System Constraints

* **Showroom Capacity:** 10 units total.
* **Warehouse Capacity:** 30 units total.
* **Restock Trigger:** 5 units are moved from warehouse to showroom when a product runs out.
* **Review Policy:** Every  days, the store orders  and  units.

---

## ⚙️ How it Works

The simulation follows these logic steps daily:

1. **Generate Demand:** Randomly determined for Product A and B.
2. **Fulfill Sales:** Units are deducted from the showroom.
3. **Internal Transfer:** If the showroom is empty, it pulls stock from the warehouse.
4. **Inventory Review:** Every 4 days (), an order is placed ().
5. **Lead Time:** Orders arrive after 1–3 days based on a probability distribution.

---

## 📊 Data Model

We used the following probability distributions to drive the simulation:

| Variable | Probabilistic Mean () |
| --- | --- |
| **Product A Demand** | 2.80 units/day |
| **Product B Demand** | 2.10 units/day |
| **Lead Time** | 1.85 days |

---

## 🎯 Key Objectives

* Calculate average ending units in inventory and showroom.
* Identify the frequency of **shortage days**.
* Validate if experimental averages match theoretical values.
* Optimize the review period () and order quantities () to minimize stockouts.

---

## 👥 Team Project

This was a collaborative project by a **team of two**, focusing on:

* **Simulation Logic:** Building the daily state-transition model.
* **Statistical Analysis:** Comparing experimental results against theoretical distributions.

---

## 🚀 Usage

1. Clone this repository.
2. Run the simulation script (e.g., `main.py` or `.xlsx`).
3. View the generated summary report for inventory performance.

---
