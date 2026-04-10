package com.teloscopy.app.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

// ─────────────────────────────────────────────────────────────────────────────
// Shimmer colour tokens (dark-theme friendly)
// ─────────────────────────────────────────────────────────────────────────────

private val ShimmerBase = Color(0xFF1E2235)
private val ShimmerHighlight = Color(0xFF2E3555)

// ─────────────────────────────────────────────────────────────────────────────
// Core shimmer composable
// ─────────────────────────────────────────────────────────────────────────────

/**
 * An animated shimmer placeholder box.
 *
 * Uses an [infiniteRepeatable] transition that sweeps a linear gradient
 * diagonally across the box to produce a "loading skeleton" effect.
 *
 * @param modifier    Additional modifiers (e.g. weight, padding).
 * @param widthDp     Explicit width; when [Dp.Unspecified] the box fills
 *                    available width.
 * @param heightDp    Height of the placeholder box.
 * @param cornerRadius Corner rounding applied via [RoundedCornerShape].
 */
@Composable
fun ShimmerBox(
    modifier: Modifier = Modifier,
    widthDp: Dp = Dp.Unspecified,
    heightDp: Dp = 20.dp,
    cornerRadius: Dp = 8.dp
) {
    val shimmerColors = listOf(ShimmerBase, ShimmerHighlight, ShimmerBase)

    val transition = rememberInfiniteTransition(label = "shimmer")
    val translateAnim by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1200, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmerTranslate"
    )

    val brush = Brush.linearGradient(
        colors = shimmerColors,
        start = Offset(translateAnim - 300f, translateAnim - 300f),
        end = Offset(translateAnim, translateAnim)
    )

    val sizeModifier = if (widthDp != Dp.Unspecified) {
        Modifier.width(widthDp)
    } else {
        Modifier.fillMaxWidth()
    }

    Box(
        modifier = modifier
            .then(sizeModifier)
            .height(heightDp)
            .clip(RoundedCornerShape(cornerRadius))
            .background(brush)
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Compound shimmer skeletons
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A shimmer "card" skeleton with a title line and several body lines.
 *
 * @param lines Number of body-text lines to show.
 */
@Composable
fun ShimmerCard(
    modifier: Modifier = Modifier,
    lines: Int = 3
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF141829))
            .padding(16.dp)
    ) {
        ShimmerBox(heightDp = 24.dp, widthDp = 150.dp)
        Spacer(modifier = Modifier.height(12.dp))
        repeat(lines) { index ->
            ShimmerBox(
                heightDp = 16.dp,
                modifier = Modifier.fillMaxWidth(if (index == lines - 1) 0.6f else 1f)
            )
            if (index < lines - 1) {
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

/**
 * A row of three shimmer "stat card" placeholders, matching the look
 * of the telomere-stat row on the results screen.
 */
@Composable
fun ShimmerStatRow(modifier: Modifier = Modifier) {
    Row(modifier = modifier.fillMaxWidth()) {
        repeat(3) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 4.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color(0xFF141829))
                    .padding(12.dp)
            ) {
                ShimmerBox(heightDp = 32.dp, widthDp = 60.dp)
                Spacer(modifier = Modifier.height(8.dp))
                ShimmerBox(heightDp = 12.dp, widthDp = 48.dp)
            }
        }
    }
}

/**
 * A full-screen shimmer skeleton that mimics the results-screen layout
 * (summary card + stat row + several content cards).
 */
@Composable
fun ShimmerResultsSkeleton(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        ShimmerCard(lines = 2)
        Spacer(modifier = Modifier.height(16.dp))
        ShimmerStatRow()
        Spacer(modifier = Modifier.height(16.dp))
        ShimmerCard(lines = 4)
        Spacer(modifier = Modifier.height(16.dp))
        ShimmerCard(lines = 3)
    }
}
