<?php
/**
 * Clone Widgets Settings Module
 * Settings > Widget Options :: Clone Widget
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       3.4
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Clone Widgets
 *
 * @since 3.4
 * @return void
 */
if( !function_exists( 'widgetopts_settings_clone' ) ):
	function widgetopts_settings_clone(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="clone">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/clone-wordpress-widgets/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Clone Widget', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Clone any widgets easily and assign them to your selected sidebar widget areas.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_clone', 69 );
endif;
?>
