import json
import pandas as pd
from django.shortcuts import render, redirect
from .models import TripDataUpload
from django.db.models.functions import TruncDate
from datetime import datetime
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required



def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            # Authenticate and log in the user
            user = form.get_user()
            login(request, user)
            return redirect('report_list')  # Redirect after login
    else:
        form = AuthenticationForm()
    return render(request, 'login_user.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)  # Log the user out
    return redirect('login_view')  # Redirect to login page after logout

@login_required(login_url="/accounts/login/")
def report_list(request):
    
    unique_dates = TripDataUpload.objects.annotate(uploaded_at_date=TruncDate('uploaded_at')) \
                                     .values('uploaded_at_date') \
                                     .distinct()
    return render(request, 'home.html', {'unique_dates':unique_dates})


@login_required
def trip_report(request, uploaded_at):
    # Parse the 'Oct. 16, 2025' format into a datetime object
    uploaded_at_date = datetime.strptime(uploaded_at, '%b. %d, %Y').date()

    # Apply TruncDate to the uploaded_at field and filter by the parsed date
    report_list = TripDataUpload.objects.annotate(uploaded_at_date=TruncDate('uploaded_at')) \
                                       .filter(uploaded_at_date=uploaded_at_date) \
                                       .distinct()

    # Pass the report list to the template
    return render(request, 'report_name.html', {'report_list': report_list, 'uploaded_at': uploaded_at})

@login_required
def daily_report_view(request, name):


    # Load files
    trip_file = TripDataUpload.objects.filter(file_type='trip', name=name).first()
    testdata_file = TripDataUpload.objects.filter(file_type='testdata').order_by('-uploaded_at').first()

    if not trip_file or not testdata_file:
        return render(request, 'tripdata/daily_report.html', {
            'error': 'Please upload both Trip and TestData files first.'
        })

    trip_cols = [
        'customer_phone_no', 'fare', 'trip_status', 'cancelled_by',
        'created_at', 'car_type', 'no_of_bids',
        'customer_created_at', 'customer_no_of_trips', 'pickup_div'
    ]
    testdata_cols = ['Testing Number']

    # ✅ Read only needed columns efficiently
    trip_df = pd.read_excel(trip_file.file.path, usecols=lambda x: x in trip_cols, engine='openpyxl')
    testdata_df = pd.read_excel(testdata_file.file.path, usecols=lambda x: x in testdata_cols, engine='openpyxl')

    # ✅ Clean phone numbers using vectorized replace
    trip_df['clean_phone'] = (
        trip_df['customer_phone_no'].astype(str)
        .str.replace(r'\s+', '', regex=True)
        .str.lstrip('0')
    )
    test_numbers = set(
        testdata_df['Testing Number'].astype(str)
        .str.replace(r'\s+', '', regex=True)
        .str.lstrip('0')
        .values
    )
    trip_df = trip_df[~trip_df['clean_phone'].isin(test_numbers)]

    # ✅ Vectorized datetime parsing
    trip_df['created_at'] = pd.to_datetime(trip_df['created_at'], errors='coerce')
    trip_df = trip_df.dropna(subset=['created_at'])
    trip_df['date'] = trip_df['created_at'].dt.date
    trip_df['weekday'] = trip_df['created_at'].dt.strftime('%A')

    # Clean strings
    trip_df['car_type'] = trip_df['car_type'].astype(str).replace(['nan', 'None'], 'Unknown').fillna('Unknown')
    trip_df['pickup_div'] = trip_df['pickup_div'].astype(str).replace(['nan', 'None'], 'Unknown').fillna('Unknown')

    car_types = sorted(trip_df['car_type'].unique())
    car_types_with_all = ['All'] + car_types
    pickup_divs = sorted(trip_df['pickup_div'].unique())
    pickup_divs_with_all = ['All'] + pickup_divs

    # ✅ Vectorized numeric conversions
    trip_df['no_of_bids'] = pd.to_numeric(trip_df['no_of_bids'], errors='coerce')
    trip_df['is_zero_bid'] = trip_df['no_of_bids'].fillna(0).eq(0)

    # ✅ Vectorized bid range categories
    trip_df['bid_range'] = pd.cut(
        trip_df['no_of_bids'],
        bins=[-1, 0, 5, 10, float('inf')],
        labels=['Zero Bid', '2-5 Bid', '6-10 Bid', '10+ Bid']
    )

    # ✅ Vectorized customer classification
    trip_df['customer_created_at'] = pd.to_datetime(trip_df['customer_created_at'], errors='coerce')
    trip_df['days_since_signup'] = (trip_df['created_at'] - trip_df['customer_created_at']).dt.days
    trip_df['customer_age_type'] = pd.Series(
        pd.cut(trip_df['days_since_signup'], bins=[-float('inf'), 30, float('inf')], labels=['New', 'Old'])
    ).fillna('Old')

    trip_df['repeat_type'] = trip_df['customer_no_of_trips'].apply(
        lambda x: 'Repeat' if pd.notna(x) and str(x).strip() not in ['', '0', 'nan'] else 'Non-Repeat'
    )

    # ✅ Precompute boolean masks to avoid recomputation inside loops
    fare_notna = trip_df['fare'].notna()
    cancelled_mask = trip_df['trip_status'] == 'Cancelled'
    trip_confirmed_mask = trip_df['trip_status'] == 'Trip Confirmed'

    # ✅ Group once
    grouped = trip_df.groupby('date', sort=True)

    def pct(val, parent):
        return f"{(val / parent * 100):.0f}%" if parent else "0%"

    totals_acc = {k: 0 for k in [
        'total', 'confirmed', 'non_confirmed', 'trip_still_confirmed',
        'completed', 'cancelled', 'cancelled_by_customer', 'cancelled_by_driver', 'zero_bid',
        'bid_2_5', 'bid_6_10', 'bid_10_plus'
    ]}

    # ✅ Predefine arrays
    dates, totalTrips, confirmedTrips, nonConfirmedTrips, completedTrips, cancelledTrips, zeroBidTrips = [], [], [], [], [], [], []
    bid2_5Trips, bid6_10Trips, bid10PlusTrips, newCustomerTrips, oldCustomerTrips, repeatTrips, nonRepeatTrips = [], [], [], [], [], [], []

    # ✅ Predefine breakdown structure (unchanged)
    metrics = ['total', 'confirmed', 'non_confirmed', 'completed', 'cancelled_from_confirmed', 'zero_bid', 'bid_2_5', 'bid_6_10', 'bid_10_plus']
    age_keys, repeat_keys = ['All', 'New', 'Old'], ['All', 'Repeat', 'Non-Repeat']
    breakdown = {
        m: {
            ct: {
                ak: {
                    rk: {pdv: [] for pdv in pickup_divs_with_all}
                    for rk in repeat_keys
                }
                for ak in age_keys
            }
            for ct in car_types_with_all
        }
        for m in metrics
    }

    # ✅ Compute all key metrics vectorized
    for day, group in grouped:
        weekday = group['weekday'].iloc[0]
        date_label = f"{weekday}, {day.strftime('%d %b %Y')}"
        dates.append(date_label)

        # --- Vectorized day-level counts ---
        total = len(group)
        confirmed = group['fare'].notna().sum()
        non_confirmed = total - confirmed
        completed = ((group['fare'].notna()) & (~group['trip_status'].isin(['Trip Confirmed', 'Cancelled']))).sum()
        cancelled_from_confirmed = ((group['fare'].notna()) & (group['trip_status'] == 'Cancelled')).sum()
        zero_bid = group['is_zero_bid'].sum()

        bid_2_5 = ((group['no_of_bids'] >= 2) & (group['no_of_bids'] <= 5)).sum()
        bid_6_10 = ((group['no_of_bids'] >= 6) & (group['no_of_bids'] <= 10)).sum()
        bid_10_plus = (group['no_of_bids'] > 10).sum()

        # store
        totalTrips.append(total)
        confirmedTrips.append(confirmed)
        nonConfirmedTrips.append(non_confirmed)
        completedTrips.append(completed)
        cancelledTrips.append(cancelled_from_confirmed)
        zeroBidTrips.append(zero_bid)
        bid2_5Trips.append(bid_2_5)
        bid6_10Trips.append(bid_6_10)
        bid10PlusTrips.append(bid_10_plus)
        newCustomerTrips.append((group['customer_age_type'] == 'New').sum())
        oldCustomerTrips.append((group['customer_age_type'] == 'Old').sum())
        repeatTrips.append((group['repeat_type'] == 'Repeat').sum())
        nonRepeatTrips.append((group['repeat_type'] == 'Non-Repeat').sum())

        # accumulate totals
        totals_acc['total'] += total
        totals_acc['confirmed'] += confirmed
        totals_acc['non_confirmed'] += non_confirmed
        totals_acc['completed'] += completed
        totals_acc['cancelled'] += cancelled_from_confirmed
        totals_acc['zero_bid'] += zero_bid
        totals_acc['bid_2_5'] += bid_2_5
        totals_acc['bid_6_10'] += bid_6_10
        totals_acc['bid_10_plus'] += bid_10_plus

        # ✅ Still loop breakdown (keep identical)
        for ct in car_types_with_all:
            base_ct = group if ct == 'All' else group[group['car_type'] == ct]
            for ak in age_keys:
                base_age = base_ct if ak == 'All' else base_ct[base_ct['customer_age_type'] == ak]
                for rk in repeat_keys:
                    base_repeat = base_age if rk == 'All' else base_age[base_age['repeat_type'] == rk]
                    for pdv in pickup_divs_with_all:
                        final_group = base_repeat if pdv == 'All' else base_repeat[base_repeat['pickup_div'] == pdv]
                        if final_group.empty:
                            vals = [0]*9
                        else:
                            vals = [
                                len(final_group),
                                final_group['fare'].notna().sum(),
                                len(final_group) - final_group['fare'].notna().sum(),
                                ((final_group['fare'].notna()) & (~final_group['trip_status'].isin(['Trip Confirmed', 'Cancelled']))).sum(),
                                ((final_group['fare'].notna()) & (final_group['trip_status'] == 'Cancelled')).sum(),
                                final_group['is_zero_bid'].sum(),
                                ((final_group['no_of_bids'] >= 2) & (final_group['no_of_bids'] <= 5)).sum(),
                                ((final_group['no_of_bids'] >= 6) & (final_group['no_of_bids'] <= 10)).sum(),
                                (final_group['no_of_bids'] > 10).sum(),
                            ]
                        (
                            breakdown['total'][ct][ak][rk][pdv].append(vals[0]),
                            breakdown['confirmed'][ct][ak][rk][pdv].append(vals[1]),
                            breakdown['non_confirmed'][ct][ak][rk][pdv].append(vals[2]),
                            breakdown['completed'][ct][ak][rk][pdv].append(vals[3]),
                            breakdown['cancelled_from_confirmed'][ct][ak][rk][pdv].append(vals[4]),
                            breakdown['zero_bid'][ct][ak][rk][pdv].append(vals[5]),
                            breakdown['bid_2_5'][ct][ak][rk][pdv].append(vals[6]),
                            breakdown['bid_6_10'][ct][ak][rk][pdv].append(vals[7]),
                            breakdown['bid_10_plus'][ct][ak][rk][pdv].append(vals[8])
                        )

    # ✅ Build daily rows (same)
    daily_rows = []
    for i, day in enumerate(grouped.groups.keys()):
        group = grouped.get_group(day)
        weekday = group['weekday'].iloc[0]
        date_label = f"{weekday}, {day.strftime('%d %b %Y')}"
        total = totalTrips[i]
        confirmed = confirmedTrips[i]
        non_confirmed = nonConfirmedTrips[i]
        completed = completedTrips[i]
        cancelled_from_confirmed = cancelledTrips[i]
        zero_bid = zeroBidTrips[i]
        trip_still_confirmed = (group['trip_status'] == 'Trip Confirmed').sum()
        cancelled_by_customer = ((group['fare'].notna()) & (group['trip_status'] == 'Cancelled') & (group['cancelled_by'] == 'Customer')).sum()
        cancelled_by_driver = ((group['fare'].notna()) & (group['trip_status'] == 'Cancelled') & (group['cancelled_by'] == 'Driver')).sum()

        daily_rows.append({
            'date': date_label,
            'total_num': total,
            'confirmed_num': confirmed,
            'completed_num': completed,
            'cancelled_num': cancelled_from_confirmed,
            'total': total,
            'confirmed': f"{confirmed} (100%)",
            'non_confirmed': f"{non_confirmed} ({pct(non_confirmed, total)})",
            'trip_still_confirmed': f"{trip_still_confirmed} ({pct(trip_still_confirmed, confirmed)})",
            'completed': f"{completed} ({pct(completed, confirmed)})",
            'cancelled_from_confirmed': f"{cancelled_from_confirmed} ({pct(cancelled_from_confirmed, confirmed)})",
            'cancelled_by_customer': f"{cancelled_by_customer} ({pct(cancelled_by_customer, cancelled_from_confirmed)})",
            'cancelled_by_driver': f"{cancelled_by_driver} ({pct(cancelled_by_driver, cancelled_from_confirmed)})",
            'zero_bid': f"{zero_bid} ({pct(zero_bid, total)})",
        })

    totals_row = {
        'date': 'TOTAL',
        'total_num': totals_acc['total'],
        'confirmed_num': totals_acc['confirmed'],
        'completed_num': totals_acc['completed'],
        'cancelled_num': totals_acc['cancelled'],
        'total': totals_acc['total'],
        'confirmed': f"{totals_acc['confirmed']} (100%)",
        'non_confirmed': f"{totals_acc['non_confirmed']} ({pct(totals_acc['non_confirmed'], totals_acc['total'])})",
        'trip_still_confirmed': f"0 (0%)",
        'completed': f"{totals_acc['completed']} ({pct(totals_acc['completed'], totals_acc['confirmed'])})",
        'cancelled_from_confirmed': f"{totals_acc['cancelled']} ({pct(totals_acc['cancelled'], totals_acc['confirmed'])})",
        'cancelled_by_customer': f"0 (0%)",
        'cancelled_by_driver': f"0 (0%)",
        'zero_bid': f"{totals_acc['zero_bid']} ({pct(totals_acc['zero_bid'], totals_acc['total'])})"
    }
    daily_rows.append(totals_row)

    # Chart payload
    chart_payload = {
        'labels': [str(x) for x in dates],
        'series': {
            'total': totalTrips,
            'confirmed': confirmedTrips,
            'non_confirmed': nonConfirmedTrips,
            'completed': completedTrips,
            'cancelled_from_confirmed': cancelledTrips,
            'zero_bid': zeroBidTrips,
            'bid_2_5': bid2_5Trips,
            'bid_6_10': bid6_10Trips,
            'bid_10_plus': bid10PlusTrips,
            'new_customer': newCustomerTrips,
            'old_customer': oldCustomerTrips,
            'repeat_customer': repeatTrips,
            'non_repeat_customer': nonRepeatTrips,
        },
        'car_types': car_types_with_all,
        'pickup_divs': pickup_divs_with_all,
        'breakdown': breakdown
    }

    chart_json = json.dumps(chart_payload, default=int)

    return render(request, 'tripdata/daily_report.html', {
        'daily_rows': daily_rows,
        'chart_json': chart_json,
        'chart_car_types': car_types,
        'pickup_divs': pickup_divs
    })







