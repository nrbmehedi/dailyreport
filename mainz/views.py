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
import os, json
from shapely.geometry import shape, Point
from rtree import index
from tqdm import tqdm
import pandas as pd
from django.conf import settings



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
        'customer_created_at', 'customer_no_of_trips', 'pickup_div',
        'pickup_lat', 'pickup_long'  # ✅ Added coordinate columns for mapping
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

    # --- NEW GEOJSON MAPPING SECTION ---


    # Load GeoJSON (loc.json)
    loc_file_path = os.path.join(settings.BASE_DIR, 'loc.json')
    if not os.path.exists(loc_file_path):
        return render(request, 'tripdata/daily_report.html', {
            'error': 'loc.json not found. Please add it in BASE_DIR.'
        })

    with open(loc_file_path, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    # Build spatial index
    features = []
    idx = index.Index()
    for pos, feature in enumerate(geojson.get('features', [])):
        geom = shape(feature['geometry'])
        features.append((geom, feature.get('properties', {})))
        idx.insert(pos, geom.bounds)

    def find_admin_info(lon, lat):
        if pd.isna(lon) or pd.isna(lat):
            return {"division": "", "district": "", "upazila": "", "union": ""}
        point = Point(float(lon), float(lat))
        for pos in idx.intersection((point.x, point.y, point.x, point.y)):
            geom, props = features[pos]
            if geom.contains(point):
                return {
                    "division": props.get("NAME_1", ""),
                    "district": props.get("NAME_2", ""),
                    "upazila": props.get("NAME_3", ""),
                    "union": props.get("NAME_4", "")
                }
        return {"division": "", "district": "", "upazila": "", "union": ""}

    # Map pickup coordinates to location
    mapped_rows = []
    for _, row in tqdm(trip_df.iterrows(), total=len(trip_df), desc="Mapping upazila"):
        lat, lon = row.get('pickup_lat'), row.get('pickup_long')
        loc = find_admin_info(lon, lat)
        row_dict = row.to_dict()
        row_dict['division_name'] = loc['division']
        row_dict['district_name'] = loc['district']
        row_dict['upazila_name'] = loc['upazila']
        row_dict['union_name'] = loc['union']
        row_dict['pickup_div'] = loc['division'] or row_dict.get('pickup_div', 'Unknown')  # ✅ fill pickup_div
        mapped_rows.append(row_dict)

    if not mapped_rows:
        return render(request, 'tripdata/daily_report.html', {
            'error': 'No matching non-confirmed live trips found.'
        })

    trip_df = pd.DataFrame(mapped_rows)
    # --- END GEOJSON MAPPING SECTION ---

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

        total = len(group)
        confirmed = group['fare'].notna().sum()
        non_confirmed = total - confirmed
        completed = ((group['fare'].notna()) & (~group['trip_status'].isin(['Trip Confirmed', 'Cancelled']))).sum()
        cancelled_from_confirmed = ((group['fare'].notna()) & (group['trip_status'] == 'Cancelled')).sum()
        zero_bid = group['is_zero_bid'].sum()
        bid_2_5 = ((group['no_of_bids'] >= 2) & (group['no_of_bids'] <= 5)).sum()
        bid_6_10 = ((group['no_of_bids'] >= 6) & (group['no_of_bids'] <= 10)).sum()
        bid_10_plus = (group['no_of_bids'] > 10).sum()

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

        totals_acc['total'] += total
        totals_acc['confirmed'] += confirmed
        totals_acc['non_confirmed'] += non_confirmed
        totals_acc['completed'] += completed
        totals_acc['cancelled'] += cancelled_from_confirmed
        totals_acc['zero_bid'] += zero_bid
        totals_acc['bid_2_5'] += bid_2_5
        totals_acc['bid_6_10'] += bid_6_10
        totals_acc['bid_10_plus'] += bid_10_plus

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

import os
import json
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from shapely.geometry import shape, Point
from rtree import index
from tqdm import tqdm
from .models import TripDataUpload


@login_required
def non_confirmed_live5min_flagged_view(request, name):
    """
    Creates upazila flag rankings and exports full data to be used in HTML.
    """

    # --- Load uploaded files ---
    trip_file = TripDataUpload.objects.filter(file_type='trip', name=name).first()
    testdata_file = TripDataUpload.objects.filter(file_type='testdata').order_by('-uploaded_at').first()

    if not trip_file or not testdata_file:
        return render(request, 'tripdata/daily_report.html', {
            'error': 'Please upload both Trip and TestData files first.'
        })

    # --- Columns we need ---
    trip_cols = [
        'booking_id', 'customer_phone_no', 'fare', 'trip_status',
        'created_at', 'cancelled_at', 'no_of_bids',
        'pickup_lat', 'pickup_long'
    ]
    testdata_cols = ['Testing Number']

    # --- Read trip & test data ---
    trip_df = pd.read_excel(trip_file.file.path, usecols=lambda x: x in trip_cols, engine='openpyxl')
    testdata_df = pd.read_excel(testdata_file.file.path, usecols=lambda x: x in testdata_cols, engine='openpyxl')

    # --- Clean phone numbers & remove test ones ---
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

    # --- Parse datetime & compute live minutes ---
    trip_df['created_at'] = pd.to_datetime(trip_df['created_at'], errors='coerce')
    trip_df['cancelled_at'] = pd.to_datetime(trip_df['cancelled_at'], errors='coerce')
    trip_df = trip_df.dropna(subset=['created_at', 'cancelled_at']).copy()
    trip_df['live_minutes'] = (trip_df['cancelled_at'] - trip_df['created_at']).dt.total_seconds() / 60.0

    # --- Filter: non-confirmed & live > 5 mins ---
    filtered_df = trip_df[trip_df['fare'].isna() & (trip_df['live_minutes'] > 5)].copy()

    # --- Flag low-bid trips ---
    filtered_df['no_of_bids'] = pd.to_numeric(filtered_df['no_of_bids'], errors='coerce').fillna(0).astype(int)
    filtered_df['flag'] = (filtered_df['no_of_bids'] < 4).astype(int)

    # --- Load GeoJSON (loc.json) ---
    loc_file_path = os.path.join(settings.BASE_DIR, 'loc.json')
    if not os.path.exists(loc_file_path):
        return render(request, 'tripdata/daily_report.html', {
            'error': 'loc.json not found. Please add it in BASE_DIR.'
        })

    with open(loc_file_path, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    # --- Build spatial index ---
    features = []
    idx = index.Index()
    for pos, feature in enumerate(geojson.get('features', [])):
        geom = shape(feature['geometry'])
        features.append((geom, feature.get('properties', {})))
        idx.insert(pos, geom.bounds)

    def find_admin_info(lon, lat):
        if pd.isna(lon) or pd.isna(lat):
            return {"division": "", "district": "", "upazila": "", "union": ""}
        point = Point(float(lon), float(lat))
        for pos in idx.intersection((point.x, point.y, point.x, point.y)):
            geom, props = features[pos]
            if geom.contains(point):
                return {
                    "division": props.get("NAME_1", ""),
                    "district": props.get("NAME_2", ""),
                    "upazila": props.get("NAME_3", ""),
                    "union": props.get("NAME_4", "")
                }
        return {"division": "", "district": "", "upazila": "", "union": ""}

    # --- Map pickup coordinates to location ---
    mapped_rows = []
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Mapping upazila"):
        lat, lon = row.get('pickup_lat'), row.get('pickup_long')
        loc = find_admin_info(lon, lat)
        row_dict = row.to_dict()
        row_dict['division_name'] = loc['division']
        row_dict['district_name'] = loc['district']  # <-- added
        row_dict['upazila_name'] = loc['upazila']
        row_dict['union_name'] = loc['union']
        mapped_rows.append(row_dict)

    if not mapped_rows:
        return render(request, 'tripdata/daily_report.html', {
            'error': 'No matching non-confirmed live trips found.'
        })

    mapped_df = pd.DataFrame(mapped_rows)

    # --- Sum flags by upazila ---
    sum_df = mapped_df.groupby('upazila_name', as_index=False)['flag'].sum().rename(columns={'flag': 'flag_sum'})
    sum_df = sum_df.sort_values(by='flag_sum', ascending=False).reset_index(drop=True)
    sum_df['order'] = range(1, len(sum_df) + 1)

    # --- Count both 0 and 1 flags per upazila ---
    flag1_df = mapped_df.groupby('upazila_name')['flag'].sum().reset_index().rename(columns={'flag': 'flag_1_count'})
    flag0_df = mapped_df.groupby('upazila_name')['flag'].apply(lambda x: (x == 0).sum()).reset_index().rename(columns={'flag': 'flag_0_count'})

    # --- Merge counts and order ---
    sum_df = sum_df.merge(flag1_df, on='upazila_name', how='left')
    sum_df = sum_df.merge(flag0_df, on='upazila_name', how='left')

    # --- Merge order back to main dataframe ---
    final_df = mapped_df.merge(
        sum_df[['upazila_name', 'order', 'flag_sum', 'flag_1_count', 'flag_0_count']],
        on='upazila_name', how='left'
    )

    # --- Prepare JS array for HTML use (with district_name) ---
    js_array = final_df.to_dict(orient='records')
    js_var_name = f"{name}_upazilaData"

    # Convert all Timestamps and NaT to strings before dumping
    def serialize_for_json(obj):
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return obj

    js_code = json.dumps(js_array, ensure_ascii=False, default=serialize_for_json)

    # --- Prepare summary ---
    top_upazilas = sum_df.head(10).to_dict(orient='records')

    # --- Render into HTML template ---
    return render(request, 'tripdata/non_confirmed_flagged.html', {
        'dataset_name': name,
        'top_upazilas': top_upazilas,
        'js_code': js_code,  # pass the JS variable as text
        'total_rows': len(final_df),
        'unique_upazilas': len(sum_df)
    })


import os
import json
import pandas as pd
from shapely.geometry import shape, Point
from rtree import index
from tqdm import tqdm
from django.conf import settings
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import TripDataUpload

@login_required
def algorithm_view(request, name, upazila):
    # === Load files ===
    required_types = ['Location', 'Bid', 'Trip', 'Driver']
    files = {}
    for ftype in required_types:
        file_obj = (
            TripDataUpload.objects
            .filter(file_type__iexact=ftype, name__icontains=name)
            .order_by('-uploaded_at')
            .first()
        )
        if file_obj:
            files[ftype] = file_obj
    missing = [ft for ft in required_types if ft not in files]
    if missing:
        return render(request, 'tripdata/algorithm.html', {
            'error': f"Missing files for: {', '.join(missing)}. Please upload all required files first."
        })

    # === Helper: clean Excel ===
    def clean_excel(path):
        df = pd.read_excel(path, engine='openpyxl')
        df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.lower()
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
        return df.dropna(how='all')

    loc_df = clean_excel(files['Location'].file.path)
    bid_df = clean_excel(files['Bid'].file.path)
    trip_df = clean_excel(files['Trip'].file.path)
    driver_df = clean_excel(files['Driver'].file.path)

    # Ensure pickup coords exist
    for col in ['pickup_lat', 'pickup_long']:
        if col not in trip_df.columns:
            trip_df[col] = None

    # --- Merge Trip + Bid ---
    trip_bid_df = pd.merge(
        bid_df,
        trip_df[['booking_id', 'pickup_lat', 'pickup_long']],
        on='booking_id',
        how='left'
    )

    # --- Merge Driver + Location ---
    merged_driver_loc = pd.merge(
        driver_df,
        loc_df,
        left_on='present_thana',
        right_on='upazila',
        how='left'
    )
    merged_driver_loc.rename(columns={'lat': 'driver_lat', 'long': 'driver_long'}, inplace=True)

    # --- Merge Trip+Bid with Driver+Location ---
    merged_final = pd.merge(
        trip_bid_df,
        merged_driver_loc,
        left_on='driver_phone_no',
        right_on='mobile',
        how='left'
    )

    # Keep relevant columns
    filtered_df = merged_final[['booking_id', 'mobile', 'driver_lat', 'driver_long', 'pickup_lat', 'pickup_long']].copy()

    # === Load loc.json and build spatial index ===
    loc_file_path = os.path.join(settings.BASE_DIR, 'loc.json')
    if not os.path.exists(loc_file_path):
        return render(request, 'tripdata/algorithm.html', {
            'error': 'loc.json not found. Please add it in BASE_DIR.'
        })

    with open(loc_file_path, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    features = []
    idx = index.Index()
    for pos, feature in enumerate(geojson.get('features', [])):
        geom = shape(feature['geometry'])
        features.append((geom, feature.get('properties', {})))
        idx.insert(pos, geom.bounds)

    def find_admin_info(lon, lat):
        if pd.isna(lon) or pd.isna(lat):
            return {"district": "", "upazila": ""}
        try:
            point = Point(float(lon), float(lat))
        except ValueError:
            return {"district": "", "upazila": ""}
        for pos in idx.intersection((point.x, point.y, point.x, point.y)):
            geom, props = features[pos]
            if geom.contains(point):
                return {
                    "district": props.get("NAME_2", ""),
                    "upazila": props.get("NAME_3", "")
                }
        return {"district": "", "upazila": ""}

    # === Map driver and pickup coordinates ===
    mapped_rows = []
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Mapping coordinates"):
        driver_loc = find_admin_info(row.get('driver_long'), row.get('driver_lat'))
        pickup_loc = find_admin_info(row.get('pickup_long'), row.get('pickup_lat'))
        row_dict = row.to_dict()
        row_dict['driver_district'] = driver_loc['district']
        row_dict['driver_upazila'] = driver_loc['upazila']
        row_dict['pickup_district'] = pickup_loc['district']
        row_dict['pickup_upazila'] = pickup_loc['upazila']
        mapped_rows.append(row_dict)

    mapped_df = pd.DataFrame(mapped_rows)
    if mapped_df.empty:
        return render(request, 'tripdata/algorithm.html', {'error': 'No valid mapped coordinates found.'})

    # === Normalize UTM upazila parameter ===
    utm_upazila_norm = upazila.strip().replace(" ", "").lower()
    mapped_df['driver_upazila_norm'] = mapped_df['driver_upazila'].astype(str).str.strip().str.replace(" ", "").str.lower()
    mapped_df['pickup_upazila_norm'] = mapped_df['pickup_upazila'].astype(str).str.strip().str.replace(" ", "").str.lower()

    # ===========================================================
    # 1️⃣ Driver-centric table
    # ===========================================================
    driver_df_filtered = mapped_df[mapped_df['driver_upazila_norm'] == utm_upazila_norm]
    driver_total_unique = driver_df_filtered['mobile'].nunique()
    driver_group = (
        driver_df_filtered
        .groupby(['pickup_district', 'pickup_upazila'])
        .agg(
            bid_count=('booking_id', 'count'),
            unique_drivers=('mobile', 'nunique')
        )
        .reset_index()
    )
    driver_group['total_unique_drivers'] = driver_total_unique
    driver_max_df = driver_group.sort_values(by=['unique_drivers', 'bid_count'], ascending=[False, False])
    driver_max_df['order'] = range(1, len(driver_max_df) + 1)

    # ===========================================================
    # 2️⃣ Pickup-centric table
    # ===========================================================
    pickup_df_filtered = mapped_df[mapped_df['pickup_upazila_norm'] == utm_upazila_norm]
    pickup_total_unique = pickup_df_filtered['mobile'].nunique()
    pickup_group = (
        pickup_df_filtered
        .groupby(['pickup_district', 'pickup_upazila', 'driver_district', 'driver_upazila'])
        .agg(
            bid_count=('booking_id', 'count'),
            unique_drivers=('mobile', 'nunique')
        )
        .reset_index()
    )
    pickup_group['total_unique_drivers'] = pickup_total_unique
    pickup_max_df = pickup_group.sort_values(by=['unique_drivers', 'bid_count'], ascending=[False, False])
    pickup_max_df['order'] = range(1, len(pickup_max_df) + 1)

    driver_table_json = driver_max_df.to_json(orient='records', force_ascii=False)
    pickup_table_json = pickup_max_df.to_json(orient='records', force_ascii=False)

    return render(request, 'tripdata/algorithm.html', {
        'driver_table_json': driver_table_json,
        'pickup_table_json': pickup_table_json,
        'upazila': upazila,
        'name': name,
    })
