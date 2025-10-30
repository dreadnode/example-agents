<?php
/**
 * Days & Date Range Settings Module
 * Settings > Widget Options :: Days & Date Range
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Days & Date Range Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */
if( !function_exists( 'widgetopts_settings_dates' ) ):
	function widgetopts_settings_dates(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="dates">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/restrict-wordpress-widgets-per-days-date-range/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Days & Date Range', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Restrict widget visibility in any day of the week and/or specific date range.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_dates', 110 );
endif;
?>
